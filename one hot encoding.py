def onehot_encoding():
    categories=[]
    encoding={}
    
    number_cat=int(input("Enter the number of categories:"))
    for _ in range(number_cat):
        category=input("Enter a category:")
        categories.append(category)
    
    diff_categories=list(set(categories))
    
    for i,category in enumerate(diff_categories):
        encoding[category]=[0]*i+[1]+[0]*(len(diff_categories)-i-1)
    encoded_vals=[encoding[category] for category in categories]
    return categories,encoded_vals

def main():
 categories,encoded_vals=onehot_encoding()
 print("One-hot encoded values are:")
 for category,encoded in zip(categories, encoded_vals):
    print(f"{category}:{encoded}")
    
main()
