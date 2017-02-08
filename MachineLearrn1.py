from sklearn import tree  # Tree Classifier
from sklearn.neighbors import KNeighborsClassifier  # another classifier  K nearest neighbour
from sklearn.linear_model import LogisticRegression

#[height, weight, shoe_size]
X = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],[159,55,33],
     [171,75,42],[181,85,43]]
Y= ['male','female','female','female','male','male','male','female','male','female','male']

clf = tree.DecisionTreeClassifier()  # from tree classifier...store decision tree classifier

clf = clf.fit(X,Y)  #Train the CLassifier with the data set ....now it knows basic characteristics of male and female and their differences

prediction = clf.predict([190,70,43,])

print(prediction)

#*****************************************************************************************************
#2nd classifier from sklearn

neigh = KNeighborsClassifier(n_neighbors=3)



X1 = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y2 = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

neigh.fit(X1, Y2)

prediction2 = neigh.predict([[190, 70, 43]])

print(prediction2)

#------------------------------------------------------------------------------------------------------------

neigh2 = LogisticRegression()

#[height, weight, shoe_size]
X3 = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y3 = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

neigh2.fit(X3, Y3)

prediction3 = neigh2.predict([[190, 70, 43]])

print(prediction3)


#---------------------------------------------------------------------------------------------------------------------

from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB()

X4 = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y4 = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

gnb = gnb.fit(X, Y)

prediction4 = gnb.predict([[190, 70, 43]])

print(prediction4)


#-----------------------------------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(n_estimators=2)

#[height, weight, shoe_size]
X5 = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y5 = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

clf2 = clf2.fit(X, Y)

prediction5 = clf2.predict([[190, 70, 43]])

print(prediction5)

#-----------------------------------------------------------------------------------------------------------------------

