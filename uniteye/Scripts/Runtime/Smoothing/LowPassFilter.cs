//Adapted from https://github.com/DarioMazzanti/OneEuroFilterUnity/blob/bb6e6a4494efce138b395d8ee21a52927e2660d4/Assets/Scripts/OneEuroFilter.cs
//
//MIT License
//
//Copyright (c) 2017 DarioMazzanti
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

using UnityEngine;
namespace UnitEye
{

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
}
