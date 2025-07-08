from math import pow,pi,exp,sqrt

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def variance(numbers):
    # s² = Σ(xi - x̄)² / (n - 1) sample variance

    mu = mean(numbers)
    distances = [pow(x - mu,2) for x in numbers]
    total = float(len(numbers))
    return sum(distances) / (total-1)

def standard_deviation(numbers):
    return sqrt(variance(numbers=numbers))

def gaussian_pdf(x,average, variance):

    exponent = exp(-(pow((x-average),2) / (2*variance)))
    return (1 / (sqrt(2*pi) *sqrt(variance))) * exponent 


def count_occurrences(items):
    
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def count_frequencies(items):
    counts = count_occurrences(items)
    total = len(items)
    return {k: v / total for k, v in counts.items()}