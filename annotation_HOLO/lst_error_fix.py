# Open the file in read mode and read lines
with open('vallistbinary.lst', 'r') as file:
    lines = file.readlines()

# Open the file in write mode to overwrite it
with open('vallistbinary.lst', 'w') as file:
    for line in lines:
        # Split the line into parts
        parts = line.split(" ")

        # Get 20 and last number from each line
        number_20 = int(parts[2])
        last_number = int(parts[-1])

        # If 20 is less than or equal to the last number, write back to the file
        if number_20 < last_number:
            file.write(line)

print("Modification is complete.")