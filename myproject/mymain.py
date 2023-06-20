import pandas as pd
import numpy as np
from tkinter import *
from tkinter import simpledialog
import tkinter as tk
from tkinter import filedialog
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn import svm
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Define the root window
root = tk.Tk()
root.geometry("600x500")
root.title("Project")
W = root.winfo_screenwidth()
H = root.winfo_screenheight()

frame1 = tk.Frame(root, bg="gray", width=800, height=500)  # ,borderwidth=3
frame1.grid(row=0, column=0)
frame1.place(x=0, y=0, width=W, height=H)

frame2 = tk.Frame(root, bg="gray", width=800, height=500)  # ,borderwidth=3
frame2.grid(row=0, column=0)
frame2.place(x=0, y=0, width=W, height=H)

# create the main frame
main = tk.Frame(root, bg="gray", width=800, height=500)  # ,borderwidth=3
main.grid(row=0, column=0)
main.place(x=0, y=0, width=W, height=H)

# create a function to find a dataset
data = None


def chooseFile():
    filepath = filedialog.askopenfilename(initialdir="‪D:\\L3.2\\Machine Learning\\SecMl",
                                          title="choose file",
                                          filetypes=(("Csv files", "*.csv"),
                                                     ("all files", "."))
                                          )
    file = open(filepath, 'r')
    global data
    data = pd.read_csv(file)
    print(data)
    print("--------------------------------------------------------")


# /\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\
def backHome():
    main = tk.Frame(root, bg="gray", width=800, height=500)  # ,borderwidth=3
    main.grid(row=0, column=0)
    main.place(x=0, y=0, width=W, height=H)
    # Create a button to select the dataset file
    btnSelect = tk.Button(main, text="Choose file", width="15", font="10", height="5", command=chooseFile, fg="black")
    btnSelect.grid(column=1, row=0, sticky="w", pady=15)
    btnSelect.place(x=50, y=30)

    # Create a button to prossing the dataset file
    btnProcessing = tk.Button(main, command=processAction, text="Processing", width="15", font="10", height="5",
                              fg="black")
    btnProcessing.grid(column=2, row=0, sticky="w", pady=15)
    btnProcessing.place(x=400, y=30)

    # Create a button to regression the dataset file
    btnRegression = tk.Button(main, text="Regression", width="15", font="10", height="5", fg="black")
    btnRegression.grid(column=1, row=1, sticky="w", pady=15)
    btnRegression.place(x=400, y=300)

    # Create a button to classification the dataset file
    btnClassification = tk.Button(main, text="Classification", width="15", font="10", height="5", fg="black")
    btnClassification.grid(column=2, row=2, sticky="w", pady=15)
    btnClassification.place(x=50, y=300)


# /\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\
def labelEncoder():
    le = LabelEncoder()
    data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
    print(data)


# /\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\//\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\

# creating classification window

def open_clf_window():
    clf_window = tk.Tk()
    clf_window.title("Classification")
    clf_window.geometry("620x420")
    clf_window.config(background="gray")

    def clss_testknn():
        x = data.iloc[:, : -1].values
        y = data.iloc[:, -1].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(x_train, y_train)

        # prediction
        predictions = model.predict(x_test)  # اختبر واداني اجابه
        matrix = confusion_matrix(y_test, predictions)
        # print(matrix)
        Label(clf_window, text=matrix).place(x=250, y=170)

        acc = accuracy_score(y_test, predictions)
        Label(clf_window, text=acc).place(x=250, y=170)

        rec = recall_score(y_test, predictions)
        Label(clf_window, text=rec).place(x=250, y=170)

        f1s = f1_score(y_test, predictions)
        Label(clf_window, text=f1s).place(x=250, y=170)

    def knn():
        global data
        x = data.iloc[:, : -1].values
        y = data.iloc[:, -1].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(x_train, y_train)

        btn_test = tk.Button(clf_window, text="Test", width=8, command=clss_testknn, state=NORMAL).place(x=170, y=305)

    def svm():
        global kernel
        global svmText, svmtxt
        svm_window = tk.Tk()
        svm_window.geometry("420x420")
        svm_window.title("SVM")
        svm_window.configure(background="gray")
        label_kernel = tk.Label(svm_window, text="Enter the type of kernel")
        svmText = tk.Text(svm_window, height=2)
        svmTestBtn = tk.Button(svm_window, command=svm_test, text="Test", width="20")
        svmTrainBtn = tk.Button(svm_window, command=svm_train, text="Train", width="20")
        svmTrainBtn.place(x=90, y=150)
        svmTestBtn.place(x=220, y=150)
        svmtxt = tk.Text(svm_window, width=40, height=10)
        svmtxt.place(x=70, y=190)
        label_kernel.pack()
        svmText.pack()
        kernel = svmText.get("1.0", END)

    def svm_train():
        global kernel
        global classifier
        global X_train, y_train, X_test, y_test
        global data
        global counter
        counter = 0
        counter = counter + 1
        data = pd.read_csv("student_scores.csv")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if (kernel == "rbf"):
            # Create an SVM classifier
            classifier = SVC(kernel='rbf', random_state=0)

        else:
            classifier = SVC(kernel='linear', random_state=0)

        # Train the model
        classifier.fit(X_train, y_train)

    def svm_test():
        global counter, accuracy, precision, recall, f1, conf_matrix, classifier
        precision = None
        recall = None
        f1 = None
        conf_matrix = None
        # Make predictions on the test set
        if (counter > 0):
            svmtxt.delete(1.0, tk.END)
            predictions = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            svmtxt.insert(tk.END, "Accuracy is ")
            svmtxt.insert(tk.END, accuracy)
            svmtxt.insert(tk.END, "\n")
            precision = precision_score(y_test, predictions, average='micro')
            svmtxt.insert(tk.END, "\n")
            svmtxt.insert(tk.END, "Precision is ")
            svmtxt.insert(tk.END, precision)
            svmtxt.insert(tk.END, "\n")
            recall = recall_score(y_test, predictions, average='micro')
            svmtxt.insert(tk.END, "\n")
            svmtxt.insert(tk.END, "Recall is ")
            svmtxt.insert(tk.END, recall)
            svmtxt.insert(tk.END, "\n")
            f1 = f1_score(y_test, predictions, average='micro')
            svmtxt.insert(tk.END, "\n")
            svmtxt.insert(tk.END, "F1 score is ")
            svmtxt.insert(tk.END, f1)
            conf_matrix = confusion_matrix(y_test, predictions)
            svmtxt.insert(tk.END, "\n")
            svmtxt.insert(tk.END, "Confusion matrix is ")
            svmtxt.insert(tk.END, conf_matrix)




        else:
            svmtxt.delete(1.0, tk.END)
            svmtxt.insert(tk.END, "You didn't train the data")

        counter = 0

    knnButton = tk.Button(clf_window, text="KNN",
                          command=knn,
                          width=10,
                          height=5,
                          background="White",
                          fg="Black"
                          )

    svmButton = tk.Button(clf_window, text="SVM",
                          command=svm,
                          width=10,
                          height=5,
                          background="White",
                          fg="Black"
                          )

    svmButton.place(x=280, y=20)
    knnButton.place(x=280, y=100)

    knntxt = tk.Text(clf_window, width=20, height=10)
    knntxt.place(x=250, y=190)


# /\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\
# destroy frameRoot and create new Frame to preprocessing data
def processAction():
    global data
    main.destroy()
    frame1 = tk.Frame(root, bg="gray", width=800, height=500)
    frame1.grid(row=0, column=0)
    frame1.place(x=0, y=0, width=W, height=H)
    # create missing value button
    missingValue_button = tk.Button(frame1, text="Missing Value", width="15", font="10", height="5", fg="black",
                                    command=missingValueActoin)  #
    missingValue_button.grid(column=1, row=0, sticky="w", pady=15)
    missingValue_button.place(x=50, y=30)
    # create labelIncoder button
    labelIncoder_button = tk.Button(frame1, text="Label Incoder", width="15", font="10", height="5", fg="black",
                                    command=labelEncoder)
    labelIncoder_button.grid(column=2, row=0, sticky="w", pady=15)
    labelIncoder_button.place(x=400, y=30)
    # create oneHotIncoder button
    oneHotIncoder_button = tk.Button(frame1, text="One-Hot-Incoder", width="15", font="10", height="5", fg="black",
                                     command=oneHotEncoder)
    oneHotIncoder_button.grid(column=1, row=1, sticky="w", pady=15)
    oneHotIncoder_button.place(x=400, y=200)
    # create Scientific Mainority Over Sampling
    sMOTE_button = tk.Button(frame1, text="SMOTE", width="15", font="10", height="5", fg="black", command=smote)
    sMOTE_button.grid(column=2, row=2, sticky="w", pady=15)
    sMOTE_button.place(x=50, y=200)
    # create button to back home
    back_button = tk.Button(frame1, text="Back", width="15", font="10", height="5", fg="black", command=backHome)
    back_button.grid(column=3, row=3, sticky="w", pady=15)
    back_button.place(x=230, y=350)


# /\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\
def mean():
    global data
    ##Drop row with any missing value
    droppedData = data.dropna()
    print(droppedData)
    print("--------------------------------------------------------")
    ##filling missing value with maen
    filledData = data.fillna(droppedData.mean(numeric_only=True))
    data = filledData
    print(data)
    print("-------------------Median------------------------------")


def median():
    global data
    ##print(data)
    ##Drop row with any missing value
    droppedData = data.dropna()
    print(droppedData)
    print("--------------------------------------------------------")
    ##filling missing value with maen
    filledData = data.fillna(droppedData.median(numeric_only=True))
    data = filledData
    print(filledData)
    print("-----------------Most Frequent------------------------")


def mostFrequent():
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer.fit(data.iloc[:, :-1])
    data.iloc[:, : -1] = imputer.transform(data.iloc[:, :-1])
    print(data)


##/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\
def missingValueActoin():
    frame2 = tk.Frame(root, bg="gray", width=800, height=500)
    frame2.grid(row=0, column=0)
    frame2.place(x=0, y=0, width=W, height=H)
    # create Mean button
    meanValue_button = tk.Button(frame2, text="Mean value", width="15", font="10", height="5", fg="black", command=mean)
    meanValue_button.grid(column=2, row=0, sticky="w", pady=15)
    meanValue_button.place(x=400, y=30)
    # create Mediam button
    medianValue_button = tk.Button(frame2, text="Median value", width="15", font="10", height="5", fg="black",
                                   command=median)
    medianValue_button.grid(column=1, row=1, sticky="w", pady=15)
    medianValue_button.place(x=400, y=200)
    # create MostFrequent
    mostFrequent_button = tk.Button(frame2, text="Most-frequent value", width="15", font="10", height="5", fg="black",
                                    command=mostFrequent)
    mostFrequent_button.grid(column=2, row=2, sticky="w", pady=15)
    mostFrequent_button.place(x=50, y=30)
    # create back
    backToFrame1_button = tk.Button(frame2, text="Back", width="15", font="10", height="5", fg="black",
                                    command=processAction)  #
    backToFrame1_button.grid(column=2, row=2, sticky="w", pady=15)
    backToFrame1_button.place(x=50, y=200)


# /\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\
def oneHotEncoder():
    global data

    cat_cols = data.select_dtypes(include=["object"]).columns
    ct = ColumnTransformer([('encoder', OneHotEncoder(), cat_cols)],
                           remainder='passthrough')
    data = pd.DataFrame(ct.fit_transform(data))
    print(data)


# /\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\

def smote():
    global data
    frame1.destroy()
    frame3 = tk.Frame(root, bg="gray", width=800, height=500)
    frame3.grid(row=0, column=0)
    frame3.place(x=0, y=0, width=W, height=H)

    def smoteAc():
        print(data)
        print(data.shape);
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        size = textSize.get("1.0", 'end-1c')
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=float(size))
        print("Before OverSampling #1= ", sum(y_train == 1));
        print("Before OverSampling #0= ", sum(y_train == 0));
        sm = SMOTE();
        x_res, y_res = sm.fit_resample(x_train, y_train)
        print("______________");
        print("After OverSampling #1= ", sum(y_res == 1));
        print("After OverSampling #0= ", sum(y_res == 0));

    # .......................................................... labelConfirmRate.config(text= val)
    ##Label to Enter the test size of SMOTE
    labelSize = Label(frame3, text="Enter Test Size", font="10", width=20, height=3, bg="white")
    labelSize.grid(row=7, column=20)
    labelSize.place(x=50, y=80)
    ##Text field to show the result of SMOTE
    textSize = Text(frame3, width=20, height=3)
    textSize.grid(row=7, column=20)
    textSize.place(x=350, y=80)
    ##Button to show the result of SMOTE
    btnSmoteResult = Button(frame3, width=20, height=3, text="Result", font="10", bg="white", command=smoteAc)  # ,
    btnSmoteResult.grid(row=7, column=20)
    btnSmoteResult.place(x=50, y=280)
    ##Button to back
    buttonBack = Button(frame3, width=20, height=3, text="Back", font="10", bg="white", command=processAction)  #
    buttonBack.grid(row=7, column=20)
    buttonBack.place(x=350, y=280)


# ..................................................................................


# REGRESSION /\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\


def regrssion_func():
    print("performing linear-regresion ..... ")
    global data
    main.destroy()
    regframe = tk.Frame(root, bg="gray", width=800, height=500)
    regframe.grid(row=0, column=0)
    regframe.place(x=0, y=0, width=W, height=H)

    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    my_test_size = 0.4
    lbl = tk.Label(root, width=20, height=1, text="ENTER TEST SIZE")
    txt = tk.Entry(root, width=30)
    lbl.place(x=50, y=30)
    txt.place(x=200, y=30)

    def reassign_func():
        my_test_size = float(txt.get())
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=my_test_size, random_state=0)
        regressor2 = LinearRegression()
        regressor2.fit(x_train, y_train)
        y_pred = regressor2.predict(x_test)
        lbl2["text"] = metrics.mean_absolute_error(y_test, y_pred)
        lbl4["text"] = metrics.mean_squared_error(y_test, y_pred)
        lbl6["text"] = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    reassignbtn = tk.Button(root, text="reassign", command=reassign_func)
    reassignbtn.place(x=400, y=30, width=100, height=20)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=my_test_size, random_state=0)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    print("Intercept = ", regressor.intercept_)
    print("cof = ", regressor.coef_)
    y_pred = regressor.predict(x_test)
    df2 = pd.DataFrame({"Actual": y_test, "predicted": y_pred})
    print(df2)
    print("Mean Absolute Error = ", metrics.mean_absolute_error(y_test, y_pred))
    print(" ")
    print("Mean Squared Error = ", metrics.mean_squared_error(y_test, y_pred))
    print(" ")
    print("Root Mean Squared Error = ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # create buttons

    lbl1 = tk.Label(regframe, text="Mean Absolute Error = ", width="15", font="10", height="5", fg="black")  #
    lbl1.grid(column=1, row=1, sticky="w", pady=15)
    lbl1.place(x=50, y=130, width=200, height=50)

    lbl2 = tk.Label(regframe, text=metrics.mean_absolute_error(y_test, y_pred), width="15", font="10", height="5",
                    fg="black")  #
    lbl2.grid(column=2, row=1, sticky="w", pady=15)
    lbl2.place(x=250, y=130, width=200, height=50)

    lbl3 = tk.Label(regframe, text="Mean Squared Error = ", width="15", font="10", height="5", fg="black")  #
    lbl3.grid(column=1, row=2, sticky="w", pady=15)
    lbl3.place(x=50, y=180, width=200, height=50)

    lbl4 = tk.Label(regframe, text=metrics.mean_squared_error(y_test, y_pred), width="15", font="10", height="5",
                    fg="black")  #
    lbl4.grid(column=2, row=2, sticky="w", pady=15)
    lbl4.place(x=250, y=180, width=200, height=50)

    lbl5 = tk.Label(regframe, text="Root Mean Squared Error =  ", width="15", font="10", height="5", fg="black")  #
    lbl5.grid(column=1, row=3, sticky="w", pady=15)
    lbl5.place(x=50, y=230, width=200, height=50)

    lbl6 = tk.Label(regframe, text=np.sqrt(metrics.mean_squared_error(y_test, y_pred)), width="15", font="10",
                    height="5",
                    fg="black")  #
    lbl6.grid(column=2, row=3, sticky="w", pady=15)
    lbl6.place(x=250, y=230, width=200, height=50)

    print("Linear-regression performed.")


# /\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\

# Create a button to select the dataset file
btnSelect = tk.Button(main, text="Choose file", width="15", font="10", height="5", command=chooseFile, fg="black")
btnSelect.grid(column=1, row=0, sticky="w", pady=15)
btnSelect.place(x=50, y=30)
# Create a button to prossing the dataset file
btnProcessing = tk.Button(main, command=processAction, text="Processing", width="15", font="10", height="5", fg="black")
btnProcessing.grid(column=2, row=0, sticky="w", pady=15)
btnProcessing.place(x=400, y=30)

# Create a button to regression the dataset file
btnRegression = tk.Button(main, command=regrssion_func, text="Regression", width="15", font="10", height="5",
                          fg="black")
btnRegression.grid(column=1, row=1, sticky="w", pady=15)
btnRegression.place(x=400, y=300)

# Create a button to classification the dataset file
btnClassification = tk.Button(main, text="Classification", width="15", font="10", height="5", fg="black",
                              command=open_clf_window)
btnClassification.grid(column=2, row=2, sticky="w", pady=15)
btnClassification.place(x=50, y=300)
# /\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\
# Openning classification window


root.mainloop()