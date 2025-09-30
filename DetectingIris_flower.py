from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris(

)
x = iris.data
y = iris .target

x_train, x_test , y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=42)


print("Dataset loaded successfully.")
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}\n")


print("Training the Logistic Regression model...")
model=LogisticRegression(max_iter=200)
model.fit(x_train,y_train)
print("model training complete.")


y_pred=model.predict(x.test)
accuracy= accuracy_score(y_test,y_pred)
print(f"model accuracy on test set:{accuracy:.2f}")


print("\ntarget classes(species):")
for i, name in enumerate(iris.target_names):
    print(f"{i}:{name}")



print("\n Example prediction:")
sample_to_predict=x_test[0].reshape(1,-1)
predicted_species_index=model.predict(sample_to_predict)[0]
predicted_species_name=iris.target_names[predicted_species_index]
actual_species_name=iris.target_names[y_test[0]]


print(f"Features of the sample: {X_test[0]}")
print(f"Predicted species: {predicted_species_name}")
print(f"Actual species: {actual_species_name}")