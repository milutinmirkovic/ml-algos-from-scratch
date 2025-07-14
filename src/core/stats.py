from math import sqrt, exp, pi, pow

def mean(nums):
    return sum(nums) / len(nums) if nums else 0.0

def variance(nums):
    n = len(nums)
    if n < 2:
        return 0.0
    mu = mean(nums)
    return sum((x - mu) ** 2 for x in nums) / (n - 1)

def gaussian_pdf(x, average, variance):
    if variance == 0:
        return 1.0 if x == average else 1e-6
    coeff   = 1.0 / (sqrt(2 * pi * variance))
    exponent = exp(-((x - average) ** 2) / (2 * variance))
    return coeff * exponent

def standarization(x):
    mu = mean(x)
    standard_dev = sqrt(variance(x))

    return (x-mu) / standard_dev