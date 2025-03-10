import os

file1 = "models/transformer_finetuned.pth"
file2 = "models/transformer_vanilla.pth"

size1 = os.path.getsize(file1)  # Size in bytes
size2 = os.path.getsize(file2)

print(f"Size of {file1}: {size1 / 1024 / 1024:.2f} MB")
print(f"Size of {file2}: {size2 / 1024 / 1024:.2f} MB")

if size1 < size2:
    print(f"{file1} is more compact.")
else:
    print(f"{file2} is more compact.")
