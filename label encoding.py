def label_encoding():
    categories=[]
    encoding_val={}
    count=0

    num_categories=int(input("Enter the number of categories: "))
    for _ in range(num_categories):
        category=input("Enter a category: ")
        categories.append(category)
        if category not in encoding_val:
            encoding_val[category]=count
            count += 1

    encoded_val=[encoding_val[cat] for cat in categories]
    return encoded_val
def main():
 encoded_val=label_encoding()
 print("Encoded values:", encoded_val)
 
main()
