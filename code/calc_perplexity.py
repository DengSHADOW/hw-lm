import math

# Ask user for cross-entropy (bits per token)
H = float(input("Enter cross-entropy H (bits per token): "))

# Calculate perplexity
perplexity = 2 ** H

# Show result
file_name = str(input("Enter the correspond file name: "))

print("\n--- Results ---")
print(f"File: {file_name}")
print(f"\nPer-word Perplexity: {perplexity:.3f}")
