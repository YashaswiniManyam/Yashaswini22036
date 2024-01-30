input_string =  input("Enter the string:")

vowels='aieouAIEOU'
consonants='bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'

vowels_count=0
consonants_count=0

for i in input_string:
    if i in vowels:
        vowels_count+=1
    elif i in consonants:
        consonants_count+=1

print(f"The given string is:{input_string}")
print(f"Vowels:{vowels_count}")
print(f"Consonants:{consonants_count}")