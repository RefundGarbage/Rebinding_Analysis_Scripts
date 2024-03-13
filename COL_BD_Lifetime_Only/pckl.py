import pickle

# Open the pickle file in binary mode (rb for reading, wb for writing)
with open('D:\\Microscopy\\SMS_BP-master2\\Final_Wt_1exp\\Track_dump.pkl', 'rb') as file:
    # Load the object from the file
    loaded_object = pickle.load(file)

# Now you can work with the loaded object
print(loaded_object)