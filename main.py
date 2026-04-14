import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def preprocessing(df):
    #Dropping null values and task_status
    df_cleaned=df.dropna(subset=["vm_id","timestamp"]).copy()
    df_cleaned.drop(columns=["task_status"],inplace=True)
    
    #Seperating numerical and categorical columns
    numeric_cols = df_cleaned.select_dtypes(include=["float64","int64"]).columns
    categorical_cols=df_cleaned.select_dtypes(include=["object"]).columns

    #Filling up null values for numerical and categorical columns
    df_cleaned.loc[:,numeric_cols]=df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
    df_cleaned.loc[:,categorical_cols]=df_cleaned[categorical_cols].fillna(df_cleaned[categorical_cols].mode().iloc[0])

    #Dropping duplicate values if any
    df_cleaned=df_cleaned.drop_duplicates()

    #Creating the is_zombie column based on the features
    cpu_thresh = df_cleaned['cpu_usage'].quantile(0.2)
    mem_thresh = df_cleaned['memory_usage'].quantile(0.2)
    net_thresh = df_cleaned['network_traffic'].quantile(0.2)
    ins_thresh = df_cleaned['num_executed_instructions'].quantile(0.2)
    eff_thresh = df_cleaned['energy_efficiency'].quantile(0.2)

    # Create conditions (each gives 1 if true, 0 if false)
    conditions = (
        (df_cleaned['cpu_usage'] <= cpu_thresh).astype(int) +
        (df_cleaned['memory_usage'] <= mem_thresh).astype(int) +
        (df_cleaned['network_traffic'] <= net_thresh).astype(int) +
        (df_cleaned['num_executed_instructions'] <= ins_thresh).astype(int) +
        (df_cleaned['energy_efficiency'] <= eff_thresh).astype(int)
    )
    # Final zombie label (at least 3 conditions true)
    df_cleaned['zombie'] = (conditions >= 3).astype(int)
    df_cleaned['zombie'].value_counts()

    #Converting the timestamp to hour and weekday for better access
    df_cleaned["timestamp"] = pd.to_datetime(df_cleaned["timestamp"])
    df_cleaned["hour"] = df_cleaned["timestamp"].dt.hour
    df_cleaned["weekday"] = df_cleaned["timestamp"].dt.weekday
    df_cleaned= df_cleaned.drop(columns=["timestamp"])

    #Dropping vm_id as its not that important
    if "vm_id" in df_cleaned.columns:
        df_cleaned=df_cleaned.drop(columns=["vm_id"])
    
    #Handling the outliers using IQR
    for col in numeric_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3-Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned[col] = df_cleaned[col].clip(lower_bound,upper_bound)

    #Recalculating the categorical colums after dropping
    categorical_cols = df_cleaned.select_dtypes(include=["object"]).columns

    #One hot encoding
    df_cleaned = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)
    return df_cleaned

def EDA(df):
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    print('\n\nExploratory Data Analysis\n')
    print('Correlation Heatmap - \n')
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CPU Usage Histogram
    sns.histplot(
        data=df,
        x='cpu_usage',
        hue='zombie',
        bins=50,
        kde=True,
        stat='density',
        common_norm=False,
        palette={0: 'green', 1: 'red'},
        ax=axes[0]
    )
    axes[0].set_title("CPU Usage: Active vs Zombie")
    axes[0].set_xlabel("CPU Usage")
    axes[0].set_ylabel("Density")

    # Network Traffic Histogram
    sns.histplot(
        data=df,
        x='network_traffic',
        hue='zombie',
        bins=50,
        kde=True,
        stat='density',
        common_norm=False,
        palette={0: 'green', 1: 'red'},
        ax=axes[1]
    )
    axes[1].set_title("Network Traffic: Active vs Zombie")
    axes[1].set_xlabel("Network Traffic")
    axes[1].set_ylabel("Density")

    plt.suptitle("Resource Usage Distribution for Active vs Zombie Servers", fontsize=14)
    plt.tight_layout()
    plt.show()

#Splitting function
def split_data(df):
    return train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

#Scaling function
def scale_data(train_df, test_df, target):
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

#Model Training
def model_train(X_train,y_train,X_test):
    #Logistic Regression
    lr_model=LogisticRegression(class_weight='balanced',max_iter=1000,random_state=42)
    lr_model.fit(X_train,y_train)

    #Random Forest 
    rf_model=RandomForestClassifier(n_estimators=100,max_depth=10, class_weight='balanced',n_jobs=-1,random_state=42)
    rf_model.fit(X_train,y_train)

    #Prediction
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    lr_pred = (lr_probs >= 0.5).astype(int)
    rf_pred = (rf_probs >= 0.5).astype(int)
    
    hybrid_probs = (lr_probs + rf_probs) / 2
    hybrid_pred = (hybrid_probs >= 0.5).astype(int)

    return lr_pred,rf_pred,hybrid_pred,lr_model,rf_model

#Performance Matrices of the Models
def performance(lr_pred,rf_pred,final_pred,y_test,model):
    #Classification Report
    print("\n\n------Classification Report------")
    print("Logistic Regression Model")
    print(classification_report(y_test,lr_pred))
    print("\n\nRandom Forest Model")
    print(classification_report(y_test,rf_pred))
    print("\n\nHybrid Model")
    print(classification_report(y_test,np.round(final_pred)))

    #Confusion Matrix
    print("\n\n------Confusion Matrix------")
    print("Logistic Regression Model")
    print(confusion_matrix(y_test,lr_pred))
    print("\n\nRandom Forest Model")
    print(confusion_matrix(y_test,rf_pred))
    print("\n\nHybrid Model")
    print(confusion_matrix(y_test,final_pred))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Logistic Regression
    ConfusionMatrixDisplay.from_predictions(y_test, lr_pred, ax=axes[0])
    axes[0].set_title("Logistic Regression")
    # Random Forest
    ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, ax=axes[1])
    axes[1].set_title("Random Forest")
    # Hybrid Model
    ConfusionMatrixDisplay.from_predictions(y_test, final_pred, ax=axes[2])
    axes[2].set_title("Hybrid Model")

    plt.suptitle("Confusion Matrices Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()

    importances = model.feature_importances_
    features = X_train.columns

    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })

    # Sort values
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10,6))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=importance_df.head(10)
    )

    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

#Main function
dataset=pd.read_csv("cloudcomputing.csv")
df=pd.DataFrame(dataset)
print("Shape before preprocessing",df.shape)
preprocessed_data=preprocessing(df)
print("Shape after preprocessing",preprocessed_data.shape)
EDA(preprocessed_data)

train_df, test_df = split_data(preprocessed_data)
X_train, X_test, y_train, y_test, scaler = scale_data(
    train_df,
    test_df,
    target="zombie"
)
lr_pred, rf_pred, final_pred, logr,randf=model_train(X_train,y_train,X_test)
performance(lr_pred,rf_pred,final_pred,y_test,randf)