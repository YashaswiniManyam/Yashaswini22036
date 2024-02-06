def euclidean_dist(v1,v2):
    if len(v1)!=len(v2):
        print("Vector sizes must have same dimensions.")
    result=0
    for i in range(len(v1)):
        result += ((v1[i]-v2[i])**2)
    return result**(0.5)

def knn_classifier(training_data,training_labels,test_instance,k):
    distances=[]
    for i,instance in enumerate(training_data):
        dist=euclidean_dist(instance,test_instance)
        distances.append((i,dist))
    distances.sort(key=lambda x:x[1])
    neighbors=distances[:k]

    class_votes={}
    for neighbor in neighbors:
        label=training_labels[neighbor[0]]
        if label in class_votes:
            class_votes[label]+=1
        else:
            class_votes[label]=1
    return max(class_votes,key=class_votes.get)

def main():
 training_data = [[1,2],[2,3],[3,4],[4,5]]
 training_labels = ['A','B','C','D']
 test_instance = [1.9,4.5]
 k = int(input("Enter the value of k: "))
 print("The training data is:",training_data)
 print("The training instance is:",training_labels)
 predicted_label = knn_classifier(training_data,training_labels,test_instance,k)
 print("Predicted label:",predicted_label)
 
main()
