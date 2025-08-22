import math

def _get_bmi(weight, height):
    try:
        if not (weight and height):
            return "Error!"
        return math.floor(int(float(weight)) / (float(height) * float(height)))
    except Exception:
        return "Error!"

def _get_breast_multiplier(bust, cup):
    try:
        bust = int(float(bust))
        bust_scale = 'Below Average' if bust < 34 else 'Above Average'
    except Exception:
        bust_scale = 'Error!'
    cup = (cup or '').upper()
    if cup == '' and bust_scale == 'Error!': return 99
    elif cup in ['AA', 'A'] and bust_scale == 'Below Average': return 1
    elif cup in ['AA', 'A'] and bust_scale == 'Above Average': return 2
    elif cup in ['B', 'C'] and bust_scale == 'Below Average': return 2
    elif cup in ['D'] and bust_scale == 'Below Average': return 3
    elif cup in ['B', 'C', 'D'] and bust_scale == 'Above Average': return 3
    elif cup in ['DD','DDD','E','EE','EEE','F','FF','G'] and bust_scale == 'Below Average': return 3
    elif cup in ['DD','DDD','E','EE','EEE','F','FF','G'] and bust_scale == 'Above Average': return 4
    elif cup in ['FFF','GG','GGG','H','HH','I'] and bust_scale == 'Below Average': return 4
    elif cup in ['FFF','GG','GGG','H','HH','I'] and bust_scale == 'Above Average': return 5
    elif cup in ['HHH','II','III','J','JJ','K'] and bust_scale == 'Below Average': return 5
    elif cup in ['HHH','II','III','J','JJ','K'] and bust_scale == 'Above Average': return 6
    else: return 0

def _get_breast_desc(mult):
    return {1:'Tiny',2:'Small',3:'Medium',4:'Large',5:'Huge',6:'Massive',99:'Error!',0:'Error!'}.get(mult,'Error!')

def _get_butt_desc(hip):
    try:
        hip = int(float(hip))
        if hip <= 32: return 'Small'
        if 33 <= hip <= 39: return 'Medium'
        if 40 <= hip <= 43: return 'Large'
        if 44 <= hip <= 47: return 'Huge'
        if hip >= 48: return 'Massive'
    except Exception:
        pass
    return 'Error!'

def _get_body_shape(bust, waist, hip):
    try:
        bust = int(float(bust)); waist = int(float(waist)); hip = int(float(hip))
        if (waist * 1.25) <= bust and (waist * 1.25) <= hip: return 'Hourglass'
        elif hip > (bust * 1.05): return 'Pear'
        elif hip < (bust / 1.05): return 'Apple'
        else:
            return 'Banana' if (max(bust, waist, hip) - min(bust, waist, hip)) <= 5 else 'Banana'
    except Exception:
        return 'Error!'

def _get_body_type(index, shape):
    try:
        index = int(index)
        if   1 <= index <= 17: t='A'
        elif 18 <= index <= 22: t='B'
        elif 23 <= index <= 28: t='C'
        elif 29 <= index <= 54: t='D'
        else: t='E'
        if shape == 'Error!': return 'Error!'
        if t=='A': return 'Skinny'
        if t=='B': return 'Petite'
        if t=='C' and shape != 'Hourglass': return 'Average'
        if t=='C' and shape == 'Hourglass': return 'Curvy'
        if t=='D' and shape == 'Banana': return 'BBW'
        if t=='D' and shape == 'Hourglass': return 'BBW - Curvy'
        if t=='D' and shape == 'Pear': return 'BBW - Bottom Heavy'
        if t=='D' and shape == 'Apple': return 'BBW - Top Heavy'
        if t=='E' and (shape in ['Banana','Hourglass']): return 'SSBBW'
        if t=='E' and shape == 'Apple': return 'SSBBW - Top Heavy'
        if t=='E' and shape == 'Pear': return 'SSBBW - Bottom Heavy'
        return 'Average'
    except Exception:
        return 'Error!'