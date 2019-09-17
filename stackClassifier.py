from imports import *

class sc:
    def __init__(self, load = False):

        '''
        * NO PARAMETERS REQUIRED FOR INITIALIZATION
        '''

        self.df = self.load_data(load)


    def drop_tags(self, tags_df, tags_to_keep):

        total_tag_names = list(tags_df.Tag)
        tags_to_drop = []
        for i in range(0, len(total_tag_names)):
            if total_tag_names[i] not in tags_to_keep:
                tags_to_drop.append(i)
        return tags_to_drop


    def remove_HTMLtags(self,text):
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text)


    def Punctuation(self, string):

        # punctuation marks
        punctuations = '''!()-[]{};:'"\,<>./?@#\n$%^&*_~\n\n'''

        # traverse the given string and if any punctuation
        # marks occur replace it with null
        for x in string.lower():
            if x in punctuations:
                string = string.replace(x, "")

        # Print string without punctuation
        return string


    def keep_token(self, t):
        return (t.is_alpha and
                not (t.is_space or t.is_punct or
                     t.is_stop or t.like_num))


    def lemmatize_doc(self, doc):
        return [ t.lemma_ for t in doc if self.keep_token(t)]


    def regex_cleanUp(self, text_doc):

        result = []
        for text in text_doc:
            x = self.remove_HTMLtags(text)
            result.append(x)
        len(result)
        text_doc = result

        result = []
        for text in tqdm.tqdm(text_doc):
            x = self.Punctuation(text)
            result.append(x)
        len(result)
        text_doc = result

        return text_doc


    def spacy_cleanUp(self, text_doc):
        doc_text = []
        for text in tqdm.tqdm(text_doc):
            doc_text.append(nlp.tokenizer(text))

        result = []
        for i in tqdm.tqdm(range(0, len(doc_text))):
            doc = self.lemmatize_doc(doc_text[i])
            result.append(doc)

        doc_text = result

        result = []
        for doc in doc_text:
            result_doc = ' '.join(doc)
            result.append(result_doc)

        doc_text = result

        return doc_text


    def drop_small_tags(self, df_questions_tags):

        total_processed_tags = df_questions_tags.Tag.value_counts().to_dict()
        small_tags_keep = []
        for key, value in total_processed_tags.items():
            if value > len(df_questions_tags) * 0.04:
                small_tags_keep.append(key)

        tags_to_drop = self.drop_tags(df_questions_tags, small_tags_keep)

        df_questions_tags.drop(tags_to_drop, inplace = True)
        df_questions_tags.reset_index(drop=True, inplace = True)

        return df_questions_tags




    def load_data(self, load):

            '''
            * NOT TO BE USED OUTSIDE OF THE CLASS
            * LOADS THE COMPLETE ACLED DATASET FROM THE DATA FOLDER
            * IT IS USED TO GET RIDE OF N/A VALUES AND UNNESSARY COLUMNS
            '''

            if load == False:

                #Import questions dataset
                questions_df = pd.read_csv('Data/Questions.csv', encoding = "ISO-8859-1")
                #Import tags dataset
                tags_df = pd.read_csv('Data/Tags.csv', encoding = "ISO-8859-1")
                #List of tags to keep
                tags_to_keep = ['pandas', 'numpy', 'matplotlib', 'list', 'dictionary', 'string',\
                                'regex', 'csv', 'arrays', 'json', 'mysql', 'scipy', 'beautifulsoup', 'xml']

                #tags to drop
                tags_to_drop = self.drop_tags(tags_df, tags_to_keep)
                tags_df.drop(tags_to_drop, inplace = True)

                #Join questions_df with tags_df
                df_questions_tags = questions_df.merge(tags_df, on = 'Id', how = 'inner')

                #Drop duplicates i.e. ids with multiple tags
                df_questions_tags.drop_duplicates('Id', keep = False, inplace = True)
                df_questions_tags.reset_index(drop=True, inplace = True)

                #Drop tags with small amount of observations
                df = self.drop_small_tags(df_questions_tags)

                #Drop unrequired columns
                columns_to_drop = ['Id','OwnerUserId', 'CreationDate', 'Score']
                df.drop(columns_to_drop, axis = 1, inplace = True)

                #regex CLEANING
                text_body = df['Body']
                text_title = df['Title']

                text_body = self.regex_cleanUp(text_body)
                text_title = self.regex_cleanUp(text_title)

                #Spacy Cleanup
                body_text = self.spacy_cleanUp(text_body)
                title_text = self.spacy_cleanUp(text_title)

                #Add processed text to dataframe

                df['body'] = body_text
                df['title'] = title_text

                pickle.dump(df, open( "processed_df.p", "wb" ))
                return df

            else:

                df = pickle.load(open( "processed_df.p", "rb" ) )
                return df



    def train_prepare_dataset(self):

        X = self.df[['body', 'title']]
        y = self.df['Tag']
        #lb = LabelBinarizer
        y = lb.fit_transform(y)
        #train test split
        X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = 0.33, random_state = 42)

        dtm_train = tfidf.fit_transform(X_train['title']+X_train['body'])

        return dtm_train, X_test, y_train, y_test, lb, tfidf





    def train_neural_network(self, model_name = None):

        #Prepare dataset for training:

        dtm_train, X_test, y_train, y_test, lb, tfidf = self.train_prepare_dataset()


        #Creating Neural Network
        tag_classifier = Sequential()
        #first layer
        tag_classifier.add(Dense(512, activation='relu', kernel_initializer='random_normal', input_dim=614843))
        tag_classifier.add(Dropout(.2))
        #second layer
        tag_classifier.add(Dense(64, activation='relu', kernel_initializer='random_normal'))
        tag_classifier.add(Dropout(.2))
        #output layer
        #softmax sums predictions to 1, good for multi-classification
        tag_classifier.add(Dense(11, activation ='softmax', kernel_initializer='random_normal'))
        tag_classifier.summary()

        #Compiling
        #adam optimizer adjusts learning rate throughout training
        #loss function categorical crossentroy for classification
        tag_classifier.compile(optimizer ='adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
        early_stop = EarlyStopping(monitor = 'loss', patience = 1, verbose = 2)

        tag_classifier.fit(dtm_train, y_train, epochs = 500,
                  batch_size = 10000, verbose = 2,
                  callbacks = [early_stop])

        if model_name != None:
            tag_classifier.save(f'{model_name}.h5')

        return tag_classifier


    def load_neural_network(self, model_name):
        tag_classifier = load_model(f'{model_name}.h5')
        return tag_classifier


    def evaluate_model_df(self, tag_classifier, c_matrix = None, c_report = None, a_score = None, load = False):
        if load:
            dtm_train, X_test, y_train, y_test, lb, tfidf = self.train_prepare_dataset()
            y_pred = pickle.load(open( "test_predictions.p", "rb" ) )
            y_predictions = lb.inverse_transform(Y = y_pred, threshold=0.5)
            y_test_result = lb.inverse_transform(Y = y_test, threshold=0)
            if c_matrix != None:
                cm = confusion_matrix(y_test_result, y_predictions)
            if c_report != None:
                c_report = classification_report(y_test_result, y_predictions)
            if a_score != None:
                a_report = accuracy_score(y_test_result, y_predictions)

            return cm, c_report, a_report
         

        else:
            dtm_train, X_test, y_train, y_test, lb, tfidf = self.train_prepare_dataset()
            dtm_test = tfidf.transform(X_test['title']+X_test['body'])
            y_pred = tag_classifier.predict(dtm_test)
            pickle.dump(y_pred, open( "test_predictions.p", "wb" ))
            y_predictions = lb.inverse_transform(Y = y_pred, threshold=0.5)
            y_test_result = lb.inverse_transform(Y = y_test, threshold=0)
            if c_matrix != None:
                cm = confusion_matrix(y_test_result, y_predictions)
            if c_report != None:
                c_report = classification_report(y_test_result, y_predictions)
            if a_score != None:
                a_report = accuracy_score(y_test_result, y_predictions)

            return cm, c_report, a_report

    def evalulate_model_text(self, tag_classifier):
        title_text = input('Enter the Title of your Stack OverFlow Question:')
        body_text = input('Enter your Stack OverFlow Question:')

        title_text = self.regex_cleanUp(title_text)
        body_text = self.regex_cleanUp(body_text)

        title_text = self.spacy_cleanUp(title_text)
        body_text = self.spacy_cleanUp(body_text)
        
        print(title_text)
        print(body_text)

        dtm_train, X_test, y_train, y_test, lb, tfidf = self.train_prepare_dataset()

        dtm_test = tfidf.transform(title_text+body_text)
        y_pred = tag_classifier.predict(dtm_test)
        y_prediction = lb.inverse_transform(Y = y_pred, threshold=0.5)

        return y_prediction
