import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.model_selection import GridSearchCV

class EmailPreprocess:
    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_selector = None
        self.feature_selection_methods = {
            'chi2': chi2,
            'f_classif': f_classif,
            'mutual_info_classif': mutual_info_classif
        }

    def load_data(self):
        self.data = pd.read_csv(self.filename)

    def check_missing_values(self):
        if self.data.isnull().sum().sum() > 0:
            self.data.fillna(0, inplace=True)

    def split_dataset(self):
        self.X = self.data.iloc[:, 1:-1]
        self.y = self.data.iloc[:, -1]

    def feature_scaling(self):
        self.scaler = MinMaxScaler()
        self.X = self.scaler.fit_transform(self.X)

    def clean_text(self):
        def text_processing(text):
            doc = nlp(text.lower())
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            return ' '.join(tokens)

        self.data['cleaned_text'] = self.data['raw_text'].apply(text_processing)

    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)

    def feature_selection(self, method='chi2', k=100):
        if method not in self.feature_selection_methods:
            raise ValueError(
                f"Метод {method} не поддерживается. Выберите из {list(self.feature_selection_methods.keys())}.")

        self.feature_selector = SelectKBest(self.feature_selection_methods[method], k=k)
        self.X_train = self.feature_selector.fit_transform(self.X_train, self.y_train)
        self.X_test = self.feature_selector.transform(self.X_test)

    def preprocess(self):
        self.load_data()
        self.check_missing_values()
        self.split_dataset()
        self.feature_scaling()
        self.train_test_split()
        self.feature_selection()

class EmailClassifier(EmailPreprocess):
    def __init__(self, filename):
        super().__init__(filename)
        self.preprocess()
        self.best_fitted_model = None

    def find_best_model_and_params(self):
        classifiers = {
            'Наивный Байес': MultinomialNB(),
            'KNN': KNeighborsClassifier(),
            'Дерево решений': DecisionTreeClassifier(),
            'Случайный лес': RandomForestClassifier(),
            'Логистическая регрессия': LogisticRegression(),
            'AdaBoost': AdaBoostClassifier(),
            'Gradient Boosting': GradientBoostingClassifier()
        }

        param_grid = {
            'Наивный Байес': {'alpha': [0.1, 0.5, 1.0]},
            'KNN': {'n_neighbors': [3, 5, 7]},
            'Дерево решений': {'max_depth': [10, 20, 30]},
            'Случайный лес': {'n_estimators': [50, 100, 150]},
            'Логистическая регрессия': {'C': [0.1, 1, 10]},
            'AdaBoost': {'n_estimators': [50, 100, 150]},
            'Gradient Boosting': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]}
        }

        highest_score = 0
        optimal_model_name = None

        for clf_name, clf in classifiers.items():
            grid_search = GridSearchCV(estimator=clf, param_grid=param_grid[clf_name], cv=5, scoring='accuracy')
            grid_search.fit(self.X_train, self.y_train)
            print(f"Лучшие параметры для {clf_name}: {grid_search.best_params_}")
            print(f"Лучшая оценка перекрестной проверки для {clf_name}: {grid_search.best_score_}")

            if grid_search.best_score_ > highest_score:
                highest_score = grid_search.best_score_
                self.best_fitted_model = grid_search.best_estimator_
                optimal_model_name = clf_name

        print(f"Оптимальная модель: {optimal_model_name} с оценкой: {highest_score}")

    def evaluate_test_metrics(self):
        if self.best_fitted_model is None:
            raise ValueError("Лучшая модель не найдена. Сначала запустите find_best_model_and_params().")

        y_pred = self.best_fitted_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def cluster_data(self, method):
        if method == 'kmeans':
            clustering_model = KMeans(n_clusters=2, random_state=42)
        elif method == 'hierarchical':
            clustering_model = AgglomerativeClustering(n_clusters=2)
        elif method == 'dbscan':
            clustering_model = DBSCAN()
        else:
            raise ValueError("Указан неподдерживаемый метод кластеризации")

        clustering_model.fit(self.X_train)
        cluster_labels = clustering_model.labels_

        cluster_0_spam_ratio = sum(cluster_labels[self.y_train == 1] == 0) / sum(cluster_labels == 0) * 100
        cluster_1_spam_ratio = sum(cluster_labels[self.y_train == 1] == 1) / sum(cluster_labels == 1) * 100

        print(f"Метод кластеризации: {method}")
        print(f"Процент спама в кластере 0: {cluster_0_spam_ratio:.2f}%")
        print(f"Процент спама в кластере 1: {cluster_1_spam_ratio:.2f}%")

        return cluster_0_spam_ratio, cluster_1_spam_ratio

if __name__ == "__main__":
    # filename = pd.read_csv('emails.csv',encoding="ISO-8859-1")
    filename = './emails.csv'
    email_classifier = EmailClassifier(filename)
    email_classifier.find_best_model_and_params()
    metrics = email_classifier.evaluate_test_metrics()
    print("Метрики на тестовом наборе данных:", metrics)
    for algo in ['kmeans', 'hierarchical', 'dbscan']:
        cluster_0_spam, cluster_1_spam = email_classifier.cluster_data(algo)
        print(f"Метод: {algo}, Процент спама в кластере 0: {cluster_0_spam:.2f}%, Процент спама в кластере 1: {cluster_1_spam:.2f}%")


