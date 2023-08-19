import pandas 

#import the class library MongoClient
from pymongo import MongoClient

#construct a new client instance of the MongoClient
mongo_client = MongoClient('localhost', 27017)# Connect to the server mongodb where you put the document collection
'''Here the server  is to be changed'''

#construct a database and collection objects instance. Verify that the collection includes documents
#new db and collection objects
'''this is to be changed'''
db = mongo_client.some_database
col = db.some_collection

#retrieve documents with an API call
#make Api call to the mongodb server using a collection objects
cursor = col.find()

#print the total number of documents with the method find()
print("total docs in collection:", col.count_documents({}))

#print the total number of documents returned in the mongodb collection
print("total docs returned by find() :", len(list(cursor)))

#get the MongoDb documents with the python list() function
mongo_docs = list(cursor)

#get only the first 50 docs in list
'''this is to be changed'''
mongo_docs = mongo_docs[:50]

#to pass the dict object
series_obj = pandas.Series({"a key": "a value"})
print("series_obj:", type(series_obj))

#create an empty Dataframe obj for storing series object
docs = pandas.DataFrame(columns=[])

#iterate over the list of MongoDB dict documents
for num, doc in enumerate(mongo_docs):
    #convert ObjectId() to str
    doc["_id"] = str(doc["_id"])  
    doc_id = doc["_id"]

'''here _id is to be changed'''


#create a Series obj from MongoDB dict
series_obj = pandas.Series(doc, name= doc_id)

#append the MongoDb series obj to the Dataframe obj
docs= docs.append(series_obj)

#export MongoDB documents to a CSV file
docs.to_csv("prefernces_list.csv", ",")


