//Source: https://github.com/DarioMazzanti/OneEuroFilterUnity/blob/bb6e6a4494efce138b395d8ee21a52927e2660d4/Assets/Scripts/OneEuroFilter.cs

using UnityEngine;

public class LowPassFilter
{
	float y, a, s;
	bool initialized;

	public void setAlpha(float _alpha)
	{
		if (_alpha <= 0.0f || _alpha > 1.0f)
		{
			Debug.LogError("alpha should be in (0.0., 1.0]");
			return;
		}
		a = _alpha;
	}

	public LowPassFilter(float _alpha, float _initval = 0.0f)
	{
		y = s = _initval;
		setAlpha(_alpha);
		initialized = false;
	}

	public float Filter(float _value)
	{
		float result;
		if (initialized)
			result = a * _value + (1.0f - a) * s;
		else
		{
			result = _value;
			initialized = true;
		}
		y = _value;
		s = result;
		return result;
	}

	public float filterWithAlpha(float _value, float _alpha)
	{
		setAlpha(_alpha);
		return Filter(_value);
	}

	public bool hasLastRawValue()
	{
		return initialized;
	}

	public float lastRawValue()
	{
		return y;
	}

}
