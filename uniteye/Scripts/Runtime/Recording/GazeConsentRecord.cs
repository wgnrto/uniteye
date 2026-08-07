using System;
using System.Security.Cryptography;

namespace UnitEye
{
    /// <summary>
    /// What a participant agreed to, written as consent.json beside the data it governs.
    ///
    /// It travels WITH the recording rather than living in a separate log, so a folder can never be found
    /// without the terms it was collected under. If this file is missing or unreadable, treat the folder as
    /// having no consent and delete it — that is the safe default, and <see cref="GazeSessionRecorder"/>
    /// refuses to record at all if it cannot write this first.
    /// </summary>
    [Serializable]
    public class GazeConsentRecord
    {
        /// <summary>Schema version of this record.</summary>
        public string recordVersion = "1";

        /// <summary>Version of the wording shown (GazeConsentTexts.WordingVersion).</summary>
        public string wordingVersion;

        /// <summary>SHA-256 of the exact consent wording shown, so the agreement can be reconstructed.</summary>
        public string wordingHash;

        /// <summary>
        /// The only participant identifier: a random token, not derived from anything about the person or
        /// the machine. Its sole job is to let someone say "delete number X" without ever having given a
        /// name. A hash of a username or device id would be a pseudonym that still links sessions across
        /// studies; a fresh random token cannot.
        /// </summary>
        public string participantToken;

        /// <summary>Tier the participant selected.</summary>
        public GazeRecordingTier tier;

        /// <summary>Whether they additionally agreed this may be published publicly.</summary>
        public bool mayPublish;

        /// <summary>
        /// Calendar date only (yyyy-MM-dd, UTC) — never the time of day, which is a movement/routine signal.
        /// Present because the deletion window is measured from it, and the consent text says so.
        /// </summary>
        public string consentedOnUtcDate;

        /// <summary>Earliest date publication is permitted; before this, withdrawal is unconditional.</summary>
        public string publicationHoldUntilUtcDate;

        /// <summary>How the participant should get in touch to withdraw. Set by the study operator.</summary>
        public string withdrawalContact;

        /// <summary>Free-text label for the study/session set, set by the operator. Never the participant.</summary>
        public string studyLabel;

        public const int PublicationHoldDays = 14;

        /// <summary>
        /// Builds a record for an affirmative consent. <paramref name="utcNow"/> is passed in rather than
        /// read here so tests are deterministic.
        /// </summary>
        public static GazeConsentRecord Create(GazeRecordingTier tier, bool mayPublish, DateTime utcNow,
            string withdrawalContact, string studyLabel)
        {
            return new GazeConsentRecord
            {
                wordingVersion = GazeConsentTexts.WordingVersion,
                wordingHash = GazeConsentTexts.WordingHash(),
                participantToken = NewParticipantToken(),
                tier = tier,
                mayPublish = mayPublish,
                consentedOnUtcDate = utcNow.ToString("yyyy-MM-dd", System.Globalization.CultureInfo.InvariantCulture),
                publicationHoldUntilUtcDate = utcNow.AddDays(PublicationHoldDays)
                    .ToString("yyyy-MM-dd", System.Globalization.CultureInfo.InvariantCulture),
                withdrawalContact = string.IsNullOrWhiteSpace(withdrawalContact) ? "(not configured)" : withdrawalContact,
                studyLabel = studyLabel ?? "",
            };
        }

        /// <summary>
        /// A short, human-transcribable token from a cryptographic RNG. Crockford-style alphabet: no I, L, O
        /// or U, so a participant reading it off the screen onto paper cannot turn it into a different valid
        /// token. System.Random would be seeded predictably enough to collide across simultaneous stations.
        /// </summary>
        public static string NewParticipantToken()
        {
            const string alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
            var bytes = new byte[12];
            using (var rng = RandomNumberGenerator.Create())
                rng.GetBytes(bytes);
            var chars = new char[14];
            int c = 0;
            for (int i = 0; i < 12; i++)
            {
                chars[c++] = alphabet[bytes[i] % alphabet.Length];
                //Grouped 4-4-4 for reliable transcription.
                if (c == 4 || c == 9) chars[c++] = '-';
            }
            return new string(chars);
        }

        /// <summary>
        /// Whether the publication hold has elapsed. The recorder cannot enforce this — it never publishes —
        /// so this exists for whatever out-of-band script does. If no such script checks it, the promise made
        /// on the consent screen is unbacked.
        /// </summary>
        public bool PublicationHoldElapsed(DateTime utcNow)
        {
            return DateTime.TryParse(publicationHoldUntilUtcDate, System.Globalization.CultureInfo.InvariantCulture,
                       System.Globalization.DateTimeStyles.AdjustToUniversal, out var until)
                   && utcNow.Date >= until.Date;
        }

        /// <summary>Whether this folder may be published at all, on its own terms.</summary>
        public bool PublishableOn(DateTime utcNow) => mayPublish && PublicationHoldElapsed(utcNow);
    }
}
