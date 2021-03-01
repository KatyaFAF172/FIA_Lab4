app = Flask(__name__)

run_with_ngrok(app)   #starts ngrok when the app is run
@app.route("/")
def home():
    uploaded = files.upload()
    df = pd.read_csv('apartmentComplexData.csv')
    column = ['value1','value2','complexAge','totalRooms','totalBedroom','complexInhabitants','apartmentsNr','value8','medianCompexValue']
    df = pd.read_csv('apartmentComplexData.csv',names=column)
    df.head()
    df.drop(['value1','value2','value8'],axis=1,inplace=True)
    df.head()
    dataset = df[['complexAge','totalRooms','totalBedroom','complexInhabitants','apartmentsNr','medianCompexValue']]
    feature_dataset = dataset.iloc[:,0:5]
    target_dataset = dataset.iloc[:,-1]
    sc_X = MinMaxScaler()
    sc_y = MinMaxScaler()
    X = sc_X.fit_transform(np.array(feature_dataset))
    y = sc_y.fit_transform(np.array(target_dataset).reshape(-1,1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
    # define the keras model
    model = Sequential()
    # Layer 1
    model.add(Dense(50, input_dim=X_train.shape[1], activation='sigmoid'))
    # Dropout regularization is added to avoid overfitting
    model.add(Dropout(0.1))
    # Layer 2
    model.add(Dense(50, activation='sigmoid'))
    # Dropout regularization is added to avoid overfitting
    model.add(Dropout(0.1))
    # Layer 3
    # Output Layer
    model.add(Dense(1))
    #Compile the model
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    model.summary()
    model.fit(X_train, y_train, batch_size=32, epochs=300)
    # prediction on training dataset
    train_prediction = model.predict(X_train)
    train_result = pd.DataFrame(sc_y.inverse_transform(y_train))
    train_result.columns = ['actual']
    train_result['predicted'] = sc_y.inverse_transform(train_prediction)
    train_result
    rms = sqrt(mean_squared_error(y_train, model.predict(X_train)))
    print ('Training Data have Root Mean Squared Error of {}'.format(rms))
    # Vizualiation (actual vs predicted) on training dataset
    fig, ax = plt.subplots()
    ax.scatter(train_result['actual'], train_result['predicted'])
    ax.plot([train_result['actual'].min(), train_result['actual'].max()], [train_result['actual'].min(), train_result['actual'].max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    plt.show()
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"
  
app.run()
