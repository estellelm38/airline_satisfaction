**L3S2 MIASHS - Introduction à l’apprentissage automatique**

**Application d’algorithmes de classification supervisée pour prédire la satisfaction**

**de passagers d’une compagnie aérienne**

**Dossier de validation**

**LONG-MERLE Estelle | No étudiant: 12106173 KAHLAOUI–GUILLAUME Rayan | No      étudiant: 12110555**

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.001.png)

Université Grenoble Alpes

Licence 3 MIASHS

**Présentation du dataset**

Le jeu de données intitulé “airline passenger satisfaction” concerne la satisfaction des passagers à bord de la compagnie aérienne suivante : United Airlines en classe Economy Plus. Il vise à comprendre les facteurs qui influencent la satisfaction des passagers lors de leurs voyages en avion.

Le jeu de données est disponible sur Kaggle : https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.002.png)

Le code utilisé pour le projet est disponible au lien github suivant : https://github.com/estellelm38/airline\_satisfaction![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.003.png)

**Data Preprocessing**

Nous cherchons à comprendre quels facteurs influencent la satisfaction des passagers aériens et quelle est leur influence respective.

L ensemble de données comprend un total de 129 880 observations et 22 variables, parmi lesquelles figurent des variables catégorielles, quantitatives et ordinales.

La target est le niveau de satisfaction du client de la compagnie aérienne dont les modalités sont : satisfied / neutral or dissatisfied.

Le jeu de données est équilibré.

Nous cherchons à obtenir la meilleure accuracy et la meilleure précision sur la target.

Les variables catégorielles comprennent des informations telles que le genre, le type de client, le type de voyage et la classe de vol. Les variables quantitatives comprennent l âge, la distance de vol, ainsi que les retards au départ et à l arrivée. Les variables ordinales représentent la satisfaction des passagers à l égard de différents services à bord, tels que le wifi à bord, la commodité des horaires de départ/arrivée, la réservation en ligne, etc.

1

Quantitatives![ref1]

Age

Flight Distance Departure Delay in Minutes

Arrival Delay in Minutes

Catégorielles

Gender (Male/Female) Customer Type (Loyal/Disloyal)

Type of Travel (Personal/Business) Class (Eco plus/Business/Eco)

Ordinales

Inflight wifi service Departure/Arrival time convenient

Ease of Online booking Gate location

Food and drink

Online boarding

Seat comfort

Inflight entertainment On-board service

Leg room service Baggage handling Checkin service

Inflight service Cleanliness

2

*Figure 1 : Features du dataset airline passenger satisfaction triées par type![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.005.png)*

Nous commençons par observer les distributions des différentes variables à l aide d histogrammes de countplots et de diagrammes pour mieux visualiser le jeu de données, les graphiques obtenus sont disponibles en annexe ( *Annexe : Figure 2* ).

Encodage

Les modèles étant basés sur des calculs mathématiques, il est nécessaire de représenter les données catégorielles et ordinales numériquement à l aide de techniques d encodage afin qu elles soient comprises par les modèles.

Nous cherchons à encoder des features, Nous allons donc utiliser les deux en- codeurs proposés par scikit learn dans ce cas de figure, à savoir one hot encoder et ordinal encoder, afin d encoder respectivement les variables catégorielles et ordinales.

Nous utilisons one hot encoder pour l encodage des variables catégorielles afin d éviter de créer des relations d ordre entre ces variables et pénaliser les modèles sensibles.

Nous allons ici décomposer les variables en sous variables et créer autant de colonnes que nous avons de catégories (celles-ci sont représentées de façon binaires). Nous obtenons une sparse matrix très large qui est automatiquement compressée.

Après encodage, 6 nouvelles colonnes ont été générées (4 pour les features et 2 pour la target).

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.006.png)

*Figure 3 : variables catégorielles une fois encodée avec le one-hot encoder* Nous utilisons l’ordinal encoder pour l encodage des variables ordinales

Il permet d associer à chaque classe une valeur numérique en traitant une par une

les différentes colonnes. Cela a permis de représenter les différentes catégories des variables ordinales par des entiers, tout en préservant l ordre des catégories.

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.007.png)

*Figure 4 : variables ordinales une fois encodée avec l’ordinal encoder*

Feature scaling : Il faut mettre toutes les features à la même échelle (par exemple l âge est de l’ordre de la dizaine et la distance est de l’ordre du millier et ne peuvent être comparés entre eux tels quels) ( *Annexe : Figure 5* )

MinMax scaling : Transformer chaque feature pour que ses valeurs soient com- prises entre 0 et 1, pour cela nous avons appliqué la formule suivante avec chaque valeur de X pour chaque feature (X-Xmin)/(Xmax-Xmin) avec X correspondant à la valeur d entrée , Xmin correspondant à la valeur minimum et Xmax la valeur maximum de la feature.

Ce processus permet de comparer les différentes features sans perdre d’information car les distances entre leurs valeurs restent proportionnelles entre elles.

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.008.png)

*Figure 6 : variables quantitatives une fois misent à l’échelle*

Nous avons également testé standard scaling et robust scaling cependant ceux- ci ont donné de moins bon résultats ce qui peut être expliqué par la grande variance des données. Pour standard scaling afin de donner un écart type de 1, celui-ci a produit plusieurs valeurs négatives qui ont par la suite complexifié les calculs de certains modèles d apprentissage. De plus, les outliers sont rares mais importants pour le jeu de données comme on le verra juste après. Ainsi

il est moins intéressant d utiliser le robust scaling qui perd de l information en utilisant un calcul de médiane plutôt qu une moyenne au profit d une robustesse face aux outliers. On va ici privilégier au contraire une méthode sensible à ces outliers.

Pour en revenir aux outliers, nous avons choisi de les garder dans le jeu de données.

En ce qui concerne les délais de départ/arrivée nous avons jugé qu’une valeur aberrante aurait une influence sur la satisfaction du passager et donc il est important de ne pas la supprimer. Ensuite nous ne supprimons pas les outliers des distances de vol parce que c est une valeur plausible (la distance maximum dans le dataset : 4983 km, est inférieure à la distance d’un vol Russie-USA : 8881 km).

Feature selection : Trouver puis sélectionner les variables les plus utiles pour notre modèle (cela permet d améliorer sa performance)

Nous avons tracé la matrice des corrélation puis avons affiché les features qui ont une corrélation |r|>0,8.

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.009.png)

*Figure 7 : paires de features corrélées à |r|>0.8*

Nous avons ensuite supprimé certaines valeurs pour garder une bonne compréhen- sion, par exemple en gardant seulement ‘satisfied’ et en supprimant ‘unsatisfied’ qui n ajoutait pas d information supplémentaire.

Nous avons supprimé une feature (le délai de départ en minutes) car c était la seule qui comportait des valeurs manquantes et ce en grande quantité (393). Ainsi nous avons préféré supprimer directement la catégorie plutôt que de simuler

ces valeurs manquantes.

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.010.png)

*Figure 8 : nombres de valeurs manquantes par features*

Nous avons ensuite plot la feature\_importance ( *Annexe : Figures 9 et 10* ) qui dépend cette fois du modèle de forêt aléatoires en observant l importance des features après entraînement du modèle. Finalement, nous avons par la suite

pu obtenir de bons résultats même en gardant toutes les features de moindre importance ainsi nous avons préféré garder toutes les features pour éviter la perte d information.

Pour la phase d apprentissage supervisé, nous avons testé les 4 classifieurs suivants : Le K-Nearest Neighbors (KNN), le Naive Bayes, l algorithme des forêts aléatoires et le SVC ainsi que la classification par un réseau de neurones (MLP).

On a décidé de représenter la qualité de l’apprentissage pour chaque modèle à l’aide d’un graphique ROC c’est une mesure de la performance pour les classificateurs binaires qui donne le taux de vrais positifs en fonction du taux de faux positifs

Pour le KNN, nous avons utilisé l estimateur KNeighborsClassifier.

Nous avons d abord divisé notre dataset en un train set (auquel nous avons décidé d attribuer 80% des données comme cela se fait généralement) et un test set (avec 20% des données).

Avec le KNN, l hyper paramètre à tester est :

Nombre de voisins (k) : Détermine le nombre de voisins à prendre en compte lors de la prédiction.

Il faut également choisir la méthode de calcul de la distance (distance euclidienne, distance de Manhattan, distance de minkowski).

Pour vérifier que notre modèle fonctionne correctement, on va utiliser la cross validation. Nous avons fait une KFold validation. Kfold permet de faire une vérification plus approfondie, plus fiable en mesurant l accuracy et la précision sur chaque Split puis en faisant la moyenne. En faisant un rééchantillonnage itératif sur les différentes valeurs de k de (de k=1 a k=10) on a pu trouver l hyper paramètre optimum (ici : k=7 avec une accuracy=0.9307822605481983 et une précision= de 97%). Cela nous permet également d entraîner et valider notre modèle sur différents splits possibles et déterminer lequel donne la meilleure performance.

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.011.png)*Figure 11 : Accuracy moyenne et erreur moyenne / standard pour une validation Kfold sur notre modèle KNN*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.012.png)

*Figure 12 : Précision moyenne pour une validation Kfold sur notre modèle KNN*

L’avantage du classifieur KNN est qu’il est plutôt simple à comprendre cependant il prend en compte chaque feature et calcul toutes les distance une a une ce qui le rend extrêmement long et sensible au bruit. Ces résultats nous servent cependant de bonne base pour l analyse.

Naive Bayes:

Les classifieurs bayésiens naïfs sont des modèles probabilistes qui cherchent à estimer la probabilité d appartenance à une classe donnée étant donné un vecteur de distribution. Ils reposent sur le théorème de Bayes pour calculer cette probabilité en se basant sur les distributions conditionnelles des features (dans notre cas : Gaussienne et Catégorielle) et la probabilité a priori des classes.

Ces classifieurs probabilistes n’ont pas d hyperparamètres, ils font l hypothèse que les features sont indépendantes les unes des autres, ce qui simplifie le calcul des probabilités conditionnelles. Pour estimer les paramètres de ces distributions,

le jeu de données est souvent découpé en fonction de la classe cible, puis les moyennes et les écarts-types (ou d autres paramètres selon la distribution choisie) sont calculés pour chaque classe.

Nous avons utilisé les estimateurs Gaussian Naive Bayes et Categorical Naive Bayes pour traiter à la fois les variables quantitatives et catégorielles.

Après séparation du dataset entre train set et test set puis cross validation, le quantitatif a obtenu une accuracy moyenne d environ 68%, tandis que le catégoriel a obtenu une accuracy moyenne d environ 89%, sur l’ensemble du dataset on obtient une accuracy totale de 82,5% et une précision de 81,6%.

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.013.png)

*Figure 14 : Accuracy et precision pour notre modèle de classification bayésien naïf*

Les avantages des classifieursbayésiens naïfs résident dans leur capacité à prendre en compte des connaissances externes pour estimer les distributions, ainsi que dans leur rapidité de calcul. Cependant, ils présentent des limites en raison de leurs suppositions de distribution et d indépendance conditionnelle.

Cet algorithme est rapide et très efficace avec un grand nombre de données donc particulièrement informatif sans réduction de dimension préalable mais il ne prend pas en compte les liens entre certaines features et est connu pour être un mauvais estimateur c’est pourquoi on ne retiendra pas les probabilités renvoyées.

Random Forests:

Les forêts aléatoires sont des modèles d apprentissage automatique qui combinent plusieurs arbres de décision pour améliorer la précision de la prédiction et réduire le surapprentissage.

Au départ, plusieurs arbres de décision sont entraînés sur des sous-ensembles aléatoires du jeu de données d entraînement. Chaque arbre est formé sur un échantillon de données tiré avec remplacement (bootstrap) et sur un sous- ensemble aléatoire des features.

Hyperparamètres à tester :

Nombre d arbres de décision à inclure dans la forêt.

Profondeur maximale des arbres : Limite la profondeur des arbres pour éviter le surapprentissage.

Nombre minimum d échantillons requis pour diviser un nœud en deux.

Chaque arbre de décision est ensuite construit en choisissant à chaque étape

le meilleur split (avec le critère de gini), parmi un sous-ensemble aléatoire des features. Cela continue jusqu à ce que les feuilles des arbres soient "pures" ou que la profondeur maximale de l arbre soit atteinte.

Pour effectuer une prédiction, chaque arbre de la forêt donne une réponse basée sur les features d entrée. Dans le cas d une classification, le résultat final est déterminé par un vote majoritaire parmi tous les arbres.

En combinant plusieurs arbres formés sur des sous-ensembles de données dif- férents, les forêts aléatoires réduisent le surapprentissage par rapport à un seul arbre de décision et augmentent la précision. Cela les rend plus robustes et moins sensibles aux variations du jeu de données d entraînement.

Les forêts aléatoires fournissent également une mesure de l importance de chaque feature dans la prédiction. Cette importance est calculée en mesurant la réduction moyenne de l impureté (par exemple, l indice de Gini) que chaque feature apporte lorsqu elle est utilisée pour diviser les arbres de la forêt.

Nous avons utilisé l estimateur RandomForestClassifier.

Nous avons donc séparé le dataset et effectué une KFold validation en testant toutes les valeurs de splits.

Nous avons obtenu une accuracy d environ 96% sur l ensemble de test avec 100 estimateurs.

Les forêts aléatoires sont robustes, peuvent gérer des données manquantes et des caractéristiques catégorielles sans prétraitement supplémentaire, et sont moins sensibles au surapprentissage par rapport aux arbres de décision simples.

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.014.png)

*Figure 16 : Accuracy et erreur moyenne pour une validation Kfold sur notre modèle de forêts aléatoires*

Dans l’ensemble, les résultats sont cohérents mais le classifieur le plus intéressant pour notre jeu de données en apprentissage supervisé semble être celui des forêts aléatoires.

SVM à noyau (appliqué à la classification : SVC) :

A pour but de séparer les données en classes en les représentant comme des vecteurs dans un espace de caractéristiques. Puis on va tracer une frontière de décision de façon à ce que la distance entre les classes et la frontière soit maximale.

Les hyperparamètres à tester sont : Le paramètre de régularisation C Le choix du noyau

Le paramètre gamma ?

Si les données ne sont pas linéairement séparables, le SVM peut utiliser une technique appelée "kernel trick" pour projeter les données dans un espace de dimension supérieure où elles peuvent être séparées linéairement.

On prend seulement les valeurs situées près de la frontière entre les classes (les vecteurs support). On mesure la distance jusqu à chacun des vecteurs support et on regarde l importance de ceux-ci mesurée dans le training, puis on trace la frontière.

Pour faire une prédiction sur un nouveau point, on va projeter ces données dans l espace de caractéristiques et les classer en fonction du côté de la frontière où ils se trouvent.

Nous avons obtenu une accuracy d environ 95% et une précision de 95% sur l ensemble de test.

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.015.png)

*Figure 18 : Accuracy et erreur moyenne pour une validation Kfold sur notre modèle SVC*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.016.png)

*Figure 19 : Precision pour une validation Kfold sur notre modèle SVC*

Le SVM est capable de trouver des frontières de décision complexes et non linéaires dans des espaces de grande dimension.

Cette méthode est intéressante dans notre cas car elle permet d atténuer le déséquilibre du jeu de données en permettant de pondérer les erreurs de classifica- tion différemment selon les classes. Ainsi, même si une classe est sous-représentée, le SVM peut encore apprendre efficacement à distinguer entre les classes.

De plus, le SVM est robuste face au surapprentissage (overfitting) et peut généraliser facilement et rapidement à de nouveaux exemples une fois entraîné.

Réseau de neurones (MLP) :

Un perceptron multicouche (MLP) est un type de réseau de neurones artificiels

qui consiste en plusieurs couches de neurones, y compris une couche d entrée, une ou plusieurs couches cachées et une couche de sortie. Chaque neurone dans le réseau est connecté à tous les neurones des couches adjacentes, formant un réseau dense. Les valeurs d entrée sont propagées à travers le réseau par des calculs pondérés et des transformations non linéaires, aboutissant à des prédictions ou

des classifications. Pendant l entraînement, les poids du réseau sont ajustés par rétropropagation du gradient pour minimiser une fonction de perte, permettant

au modèle d apprendre à partir des données.

Les hyperparamètres, tels que le nombre de couches cachées, le nombre de neurones par couche et la fonction d activation, influencent les performances et la capacité du modèle à généraliser.

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.017.png)

*Figure 20 : Hyperparamètres testés sur notre MLP*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.018.png)

*Figure 21 : Hyperparamètres pour obtenir la meilleure précision pour notre MLP*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.019.png)

*Figure 22 : Hyperparamètres pour obtenir la meilleure accuracy pour notre MLP* Pour l algorithme non supervisé, on a décidé de partir sur le modèle CAH.

On plot un dendrogramme ( *Annexe : Figure 23* ) avec tous les individus regroupés en différents clusters avec toutes les étapes de regroupement pour chaque distance

de manhattan en utilisant le complete linkage qui vise à regrouper les features

en clusters en fonction de leurs valeurs de distance les plus éloignées, puis on choisit un seuil de distance en observant le graphique. On opte pour le seuil 36 qui nous laisse 6 clusters tout en gardant une grande distance.

PCA :

Réduction de dimension : réduire la complexité d un dataset en projetant ses données dans un espace de plus petite dimension (avec moins de variables)

But : accélérer l apprentissage de la machine et lutter contre le fléau de la dimension (risque d overfitting lié au surplus de dimensions)

Principe PCA : projeter nos données sur des axes (composantes principales) en cherchant à minimiser la distance entre points et projections

Ainsi, on réduit la dimension du dataset tout en préservant au maximum la variance de nos données.

Afin de minimiser les risques d overfitting lié au surplus de dimensions, nous avons décidé de faire une réduction de dimension en utilisant le traitement PCA qui va permettre de réduire au maximum la dimension du dataset tout en préservant au maximum la variance (l information) de notre dataset.

Nous avons tout d abord calculé le pourcentage de variance expliquée puis nous avons tracé le graphique ( *Annexe : Figure 24* ) afin de mieux visualiser le rapport entre la variance expliquée et le nombre de composantes. Le but étant de

déterminer une valeur de variance la plus élevée possible avec un minimum de composantes restantes. On remarque facilement sur le graphique obtenu qu une variance de 90% avec 10 composantes semble optimale. Ce qu on a vérifié par la suite en testant différents pourcentage de variance et en comparant la précision obtenue après apprentissage supervisé sans l étape de réduction de dimension. Une observation qui saute aux yeux est la différence de vitesse de calcul qui est extrêmement plus rapide avec un traitement PCA préalable

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.020.png)

*Figure 25 : Accuracy et erreur moyenne pour une validation Kfold sur notre modèle KNN avec PCA*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.021.png)

*Figure 26 : Précision pour une validation Kfold sur notre modèle KNN avec PCA*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.022.png)

*Figure 28 : Accuracy et precision pour notre modèle de classification bayésien naïf avec PCA*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.023.png)

*Figure 30 : Accuracy en train et en test pour notre forêt aléatoire avec PCA* Voici un résumé des classifieurs testés :![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.024.png)

Classifieurs Accuracy Précision Classement Classement

Accuracy Précision KNN 0.93 0.97 3 2

Naïf Bayésien 0.82 0.82 6![ref1]

Forêts 0.96 **1**

Aléatoires

SVC 0.95 0.95 2 4 MLP 0.93 0.98 3 **1** Avec PCA

KNN 0.92 0.96 4 3 Naïf Bayésien 0.83 0.84 5 5 Forêts 0.93 3

Aléatoires![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.025.png)

Dans l étude comparative des performances des modèles de classificationappliqués

à notre ensemble de données, les résultats démontrent que le modèle de forêts aléatoires offrela meilleure précision globale, mesurée par l accuracy. En revanche, le perceptron multicouche (MLP) se distingue par sa capacité à produire la meilleure précision, en particulier dans la prédiction des classes positives, comme

l illustre le score de précision. Ces conclusions sont d autant plus significatives lorsqu on les compare aux résultats antérieurs disponibles sur Kaggle, mettant en évidence l amélioration de la performance des modèles sur notre jeu de données spécifique.

On pourrait poursuivre ce projet en aboutissant l’apprentissage non supervisé notamment en testant la méthode kmeans et en faisant l’évaluation de nos clusters avec CAH.

**Annexe![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.026.png)**

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.027.png)

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.028.png)

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.029.png)

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.030.png)

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.031.png)

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.032.png)

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.033.png)

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.034.png)

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.035.png)

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.036.png)

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.037.png)

*Figure 2 : répartition des individus par features dans le dataset![ref1]![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.038.png)*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.039.png)

*Figure 5 : description des statistiques univariées pour l’âge et la distance de vol*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.040.jpeg)

*Figure 9 : Importance de chaque feature calculée par le classifieur forêts aléatoires*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.041.jpeg)

*Figure 10 : Importances cumulées des features selon le classifieur forêts aléatoires avec un seuil à 80%*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.042.jpeg)

*Figure 13 : Graphique ROC pour KNN évalué par Kfold*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.043.jpeg)

*Figure 15 : Graphique ROC pour classifieur bayésien naïf évalué par Kfold*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.044.jpeg)

*Figure 17 : Graphique ROC pour forêts aléatoires évaluées par Kfold*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.045.jpeg)

*Figure 23 : Dendrogramme généré par CAH*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.046.jpeg)

*Figure 24 : Diagramme du coude pour la PCA avec un seuil 0.95*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.047.jpeg)

*Figure 27 : Graphique ROC pour KNN avec PCA évalué par Kfold*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.048.jpeg)

*Figure 29 : Graphique ROC pour classifieur bayésien naïf avec PCA évalué par*

*Kfold*

![](Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.049.jpeg)

*Figure 29 : Graphique ROC pour forêts aléatoires avec PCA évaluées par Kfold*
23

[ref1]: Aspose.Words.64839b93-4eac-4803-95cb-7288c9b3cbec.004.png
