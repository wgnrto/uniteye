using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;
using UnityEngine.Networking;

namespace UnitEye.EditorTools
{
    /// <summary>
    /// Uploads a packaged recording to a GitHub Release as an asset, from the Editor.
    ///
    /// Deliberately lives in the EDITOR assembly, not under Scripts/Runtime/Recording: a smoke test scans
    /// that folder for networking code so the consent screen's "we never send anything over the internet"
    /// stays literally true of the runtime a participant runs, and so no credential can reach a shipped build.
    ///
    /// The token is read from an environment variable at the moment of use and never stored, never serialized
    /// and never written to a log, an exception message or the progress UI. A serialized field would end up
    /// in a committed .asset; that is the mistake this design exists to prevent.
    ///
    /// Uploads land as a DRAFT release. Draft assets are visible only to accounts with push access, so the
    /// gap between "bytes uploaded" and "world-readable" stays under your control and a mistake is deletable.
    /// Draft is not private and not encrypted — it is on GitHub's servers and visible to every collaborator.
    /// </summary>
    public static class GitHubReleaseUploader
    {
        public const string TokenEnvVar = "UNITEYE_GITHUB_TOKEN";
        private const string ApiRoot = "https://api.github.com";
        private const string ApiVersion = "2022-11-28";
        private const string UserAgent = "UnitEye-RecordedSessionBrowser/1.0";
        //GitHub rejects release assets above this; surface it before a long upload fails.
        private const long MaxAssetBytes = 2L * 1024 * 1024 * 1024;

        public class Target
        {
            public string Owner, Repo, Tag, ReleaseTitle;
        }

        public class Result
        {
            public bool Ok;
            public string Error;
            public long AssetId;
            public long ReleaseId;
            public string AssetName;
            public string BrowserDownloadUrl;
            public string PublishedAsUser;
        }

        public static bool HasToken => !string.IsNullOrEmpty(ReadToken());

        //Read at point of use, never cached in a static: a cached credential outlives the operation that
        //needed it and shows up in memory dumps and domain-reload state for no benefit.
        private static string ReadToken()
        {
            try { return Environment.GetEnvironmentVariable(TokenEnvVar); }
            catch { return null; }
        }

        /// <summary>
        /// Confirms who the token belongs to. Publishing biometric data to the wrong account is unrecoverable
        /// and easy to do, so the operator sees the login before anything is sent.
        /// </summary>
        public static bool TryIdentify(out string login, out string error)
        {
            login = null;
            var token = ReadToken();
            if (string.IsNullOrEmpty(token))
            {
                error = $"No token. Set the {TokenEnvVar} environment variable to a GitHub token with " +
                        "Contents: Read and write on the target repository, then restart Unity.";
                return false;
            }
            using (var req = Get($"{ApiRoot}/user", token))
            {
                if (!Send(req, "Checking token…", out error)) return false;
                login = ExtractString(req.downloadHandler.text, "login");
                return true;
            }
        }

        /// <summary>
        /// Uploads <paramref name="zipPath"/> as an asset on a DRAFT release for <paramref name="target"/>.
        /// Creates the release if it does not exist. Does not publish it — that is a separate, explicit step.
        /// </summary>
        public static Result Upload(Target target, string zipPath, string releaseBody)
        {
            var result = new Result();
            var token = ReadToken();
            if (string.IsNullOrEmpty(token)) { result.Error = $"{TokenEnvVar} is not set."; return result; }

            var file = new FileInfo(zipPath);
            if (!file.Exists) { result.Error = $"Not found: {zipPath}"; return result; }
            if (file.Length > MaxAssetBytes)
            {
                result.Error = $"{FormatBytes(file.Length)} exceeds GitHub's {FormatBytes(MaxAssetBytes)} " +
                               "per-asset limit. Package fewer sessions per zip.";
                return result;
            }

            try
            {
                if (!TryIdentify(out var login, out var idError)) { result.Error = idError; return result; }
                result.PublishedAsUser = login;

                //Resolve the release, creating a draft if the tag has none yet.
                string releaseJson;
                using (var req = Get($"{ApiRoot}/repos/{target.Owner}/{target.Repo}/releases/tags/{Uri.EscapeDataString(target.Tag)}", token))
                {
                    bool found = Send(req, "Looking up release…", out var lookupError);
                    if (found) releaseJson = req.downloadHandler.text;
                    else if (req.responseCode == 404) releaseJson = null;
                    else { result.Error = lookupError; return result; }
                }

                if (releaseJson == null)
                {
                    var body = "{" +
                        $"\"tag_name\":{Quote(target.Tag)}," +
                        $"\"name\":{Quote(string.IsNullOrEmpty(target.ReleaseTitle) ? target.Tag : target.ReleaseTitle)}," +
                        $"\"body\":{Quote(releaseBody ?? "")}," +
                        "\"draft\":true,\"prerelease\":false}";
                    using (var req = PostJson($"{ApiRoot}/repos/{target.Owner}/{target.Repo}/releases", token, body))
                    {
                        if (!Send(req, "Creating draft release…", out var createError))
                        {
                            //404 here means the repo is missing OR the token cannot see it — the two are
                            //indistinguishable by design, so say both rather than guess.
                            result.Error = req.responseCode == 404
                                ? $"Repository {target.Owner}/{target.Repo} not found, or the token has no access to it."
                                : createError;
                            return result;
                        }
                        releaseJson = req.downloadHandler.text;
                    }
                }

                result.ReleaseId = ExtractLong(releaseJson, "id");
                var uploadUrl = ExtractString(releaseJson, "upload_url");
                if (string.IsNullOrEmpty(uploadUrl)) { result.Error = "Release carried no upload_url."; return result; }

                //upload_url is server-supplied and decides where a token plus facial imagery gets sent, so
                //validate the host rather than trusting it.
                int brace = uploadUrl.IndexOf('{');
                if (brace >= 0) uploadUrl = uploadUrl.Substring(0, brace);
                if (!uploadUrl.StartsWith("https://uploads.github.com/", StringComparison.Ordinal))
                {
                    result.Error = $"Refusing to upload: unexpected upload host in {uploadUrl}";
                    return result;
                }

                //Constrain the asset name rather than escaping creatively: GitHub renames assets containing
                //special characters, which would break the receipt's download URL.
                var assetName = SafeAssetName(Path.GetFileName(zipPath));
                result.AssetName = assetName;
                var url = $"{uploadUrl}?name={Uri.EscapeDataString(assetName)}";

                using (var req = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST))
                {
                    //UploadHandlerFile streams from disk; UploadHandlerRaw would hold the whole archive in
                    //managed memory and copy it to native, which for an imagery tier is a gigabyte for nothing.
                    req.uploadHandler = new UploadHandlerFile(zipPath);
                    req.uploadHandler.contentType = "application/zip";
                    req.downloadHandler = new DownloadHandlerBuffer();
                    ApplyHeaders(req, token);
                    req.redirectLimit = 0;
                    req.timeout = 0;

                    if (!Send(req, $"Uploading {FormatBytes(file.Length)}…", out var uploadError, showBytes: true))
                    {
                        result.Error = req.responseCode == 422
                            ? $"An asset named '{assetName}' already exists on release '{target.Tag}'. " +
                              "Delete it on GitHub or rename the zip, then retry."
                            : uploadError;
                        return result;
                    }

                    var assetJson = req.downloadHandler.text;
                    result.AssetId = ExtractLong(assetJson, "id");
                    result.BrowserDownloadUrl = ExtractString(assetJson, "browser_download_url");
                    var uploadedSize = ExtractLong(assetJson, "size");
                    if (uploadedSize > 0 && uploadedSize != file.Length)
                    {
                        result.Error = $"Size mismatch: sent {file.Length} B, GitHub recorded {uploadedSize} B. " +
                                       "Delete the asset and retry.";
                        return result;
                    }
                }

                result.Ok = true;
                return result;
            }
            catch (Exception e)
            {
                //Never surface the exception object verbatim: request objects can carry the Authorization
                //header, and this string reaches the console and the UI.
                result.Error = Redact(e.Message);
                return result;
            }
            finally { EditorUtility.ClearProgressBar(); }
        }

        /// <summary>Flips a draft release public. Separate call, separate confirmation — this is the irreversible one.</summary>
        public static bool Publish(Target target, long releaseId, out string error)
        {
            var token = ReadToken();
            if (string.IsNullOrEmpty(token)) { error = $"{TokenEnvVar} is not set."; return false; }
            using (var req = new UnityWebRequest(
                $"{ApiRoot}/repos/{target.Owner}/{target.Repo}/releases/{releaseId}", "PATCH"))
            {
                req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes("{\"draft\":false}"));
                req.uploadHandler.contentType = "application/json";
                req.downloadHandler = new DownloadHandlerBuffer();
                ApplyHeaders(req, token);
                return Send(req, "Publishing release…", out error);
            }
        }

        /// <summary>Deletes an uploaded asset. This is how a withdrawal request is honoured after publication.</summary>
        public static bool DeleteAsset(Target target, long assetId, out string error)
        {
            var token = ReadToken();
            if (string.IsNullOrEmpty(token)) { error = $"{TokenEnvVar} is not set."; return false; }
            using (var req = new UnityWebRequest(
                $"{ApiRoot}/repos/{target.Owner}/{target.Repo}/releases/assets/{assetId}", "DELETE"))
            {
                req.downloadHandler = new DownloadHandlerBuffer();
                ApplyHeaders(req, token);
                return Send(req, "Deleting asset…", out error);
            }
        }

        /// <summary>
        /// Records where a session was published, in the session's own folder. Without this map a withdrawal
        /// request cannot be honoured — you would know someone wants out but not which asset is theirs.
        /// Contains no credential.
        /// </summary>
        public static void WriteReceipt(string sessionFolder, Target target, Result r, DateTime utcNow)
        {
            try
            {
                var json = "{" +
                    $"\"owner\":{Quote(target.Owner)},\"repo\":{Quote(target.Repo)},\"tag\":{Quote(target.Tag)}," +
                    $"\"releaseId\":{r.ReleaseId},\"assetId\":{r.AssetId},\"assetName\":{Quote(r.AssetName)}," +
                    $"\"browserDownloadUrl\":{Quote(r.BrowserDownloadUrl)}," +
                    $"\"publishedOnUtcDate\":{Quote(utcNow.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture))}}}";
                File.WriteAllText(Path.Combine(sessionFolder, "publication-receipt.json"), json, new UTF8Encoding(false));
            }
            catch (Exception e) { UnitEyeLog.Exception(e); }
        }

        #region plumbing

        private static UnityWebRequest Get(string url, string token)
        {
            var req = UnityWebRequest.Get(url);
            ApplyHeaders(req, token);
            return req;
        }

        private static UnityWebRequest PostJson(string url, string token, string json)
        {
            var req = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST);
            req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
            req.uploadHandler.contentType = "application/json";
            req.downloadHandler = new DownloadHandlerBuffer();
            ApplyHeaders(req, token);
            return req;
        }

        private static void ApplyHeaders(UnityWebRequest req, string token)
        {
            req.SetRequestHeader("Accept", "application/vnd.github+json");
            req.SetRequestHeader("Authorization", "Bearer " + token);
            req.SetRequestHeader("X-GitHub-Api-Version", ApiVersion);
            req.SetRequestHeader("User-Agent", UserAgent);
        }

        private static bool Send(UnityWebRequest req, string label, out string error, bool showBytes = false)
        {
            error = null;
            var op = req.SendWebRequest();
            //Blocks the Editor with a cancelable bar. Honest for a maintainer action that the operator is
            //sitting and waiting on, and far simpler to reason about than an update-pumped state machine.
            while (!op.isDone)
            {
                var pct = showBytes ? req.uploadProgress : 0.5f;
                if (EditorUtility.DisplayCancelableProgressBar("GitHub", label, pct))
                {
                    req.Abort();
                    error = "Cancelled.";
                    return false;
                }
                System.Threading.Thread.Sleep(50);
            }
            EditorUtility.ClearProgressBar();

            if (req.result == UnityWebRequest.Result.Success && req.responseCode < 300) return true;

            switch (req.responseCode)
            {
                case 401: error = "401 Unauthorized - the token is invalid or expired."; break;
                case 403: error = "403 Forbidden - the token lacks Contents: write on this repository, or you are rate limited."; break;
                case 404: error = "404 Not Found - repository, tag or release does not exist, or the token cannot see it."; break;
                case 410: error = $"410 Gone - the pinned GitHub API version ({ApiVersion}) was retired; update ApiVersion."; break;
                case 422: error = "422 Unprocessable - GitHub rejected the request (an asset of that name may already exist)."; break;
                default:
                    error = req.responseCode > 0
                        ? $"HTTP {req.responseCode}: {Redact(Truncate(req.downloadHandler?.text, 300))}"
                        : $"Network error: {Redact(req.error)}";
                    break;
            }
            return false;
        }

        //Belt and braces. Nothing here should ever contain the token, but this string reaches the console.
        private static string Redact(string s)
        {
            if (string.IsNullOrEmpty(s)) return s;
            var token = ReadToken();
            if (!string.IsNullOrEmpty(token)) s = s.Replace(token, "***");
            return s;
        }

        private static string Truncate(string s, int n)
            => string.IsNullOrEmpty(s) ? "" : (s.Length <= n ? s : s.Substring(0, n) + "…");

        public static string SafeAssetName(string name)
        {
            var sb = new StringBuilder(name.Length);
            foreach (var c in name)
                sb.Append(char.IsLetterOrDigit(c) || c == '.' || c == '_' || c == '-' ? c : '-');
            return sb.ToString();
        }

        private static string Quote(string s)
            => "\"" + (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n") + "\"";

        private static string ExtractString(string json, string key)
        {
            var needle = $"\"{key}\":\"";
            int i = json != null ? json.IndexOf(needle, StringComparison.Ordinal) : -1;
            if (i < 0) return "";
            int start = i + needle.Length, end = start;
            while (end < json.Length && json[end] != '"') { if (json[end] == '\\') end++; end++; }
            return end <= json.Length ? json.Substring(start, end - start) : "";
        }

        private static long ExtractLong(string json, string key)
        {
            var needle = $"\"{key}\":";
            int i = json != null ? json.IndexOf(needle, StringComparison.Ordinal) : -1;
            if (i < 0) return 0;
            int start = i + needle.Length, end = start;
            while (end < json.Length && (char.IsDigit(json[end]) || json[end] == '-')) end++;
            return long.TryParse(json.Substring(start, end - start), NumberStyles.Integer,
                CultureInfo.InvariantCulture, out var v) ? v : 0;
        }

        private static string FormatBytes(long b)
        {
            if (b >= 1L << 30) return $"{b / (float)(1L << 30):F1} GB";
            if (b >= 1L << 20) return $"{b / (float)(1L << 20):F1} MB";
            return $"{b / (float)(1L << 10):F0} KB";
        }

        #endregion
    }
}
