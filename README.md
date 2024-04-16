# Airline Satisfaction
Airline satisfaction is a machine learning project using a dataset .csv.

## Todo :
for unsupervised learning :
- Data preprocessing 
  * Encoding variables :heavy_check_mark:
  * Imputing missing Values :heavy_check_mark:
  * Feature scaling :heavy_check_mark:
  * Outlier removals :heavy_check_mark:
  * Feature selection :heavy_check_mark:
  * Dimensionality reduction :heavy_check_mark:

- Fit to different models
- Evaluate the models
- Conclude

## The features are the following :
- Gender
- Customer Type
- Age
- Type of Travel
- Class
- Flight Distance
- Inflight wifi service
- Departure/Arrival time convenient
- Ease of Online booking
- Gate location
- Food and drink
- Online boarding
- Seat comfort
- Inflight entertainment
- On-board service
- Leg room service
- Baggage handling
- Checkin service
- Inflight service
- Cleanliness
- Departure Delay in Minutes
- Arrival Delay in Minutes
- Satisfaction

- \documentclass[11pt]{report}
\usepackage{longtable}
\usepackage{array}
\usepackage{booktabs}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{setspace}

\usepackage[default]{cmbright}
\usepackage[french]{babel}
\usepackage{tabularx}
\usepackage[svgnames]{xcolor} % Using colors
\usepackage{fancyhdr} % Needed to define custom headers/footers
\usepackage[a4paper]{geometry}  % Changing size of document
\usepackage{braket}
\usepackage[normalem]{ulem}
\usepackage[style=verbose-ibid,backend=biber]{biblatex}
\addbibresource{bibliographie.bib}

\DefineBibliographyStrings{french}{in={dans},
inseries={dans},
volume = {vol\adddot},%
volumes = {vol\adddot},%
jourvol = {vol\adddot},
}%}


% \usepackage[compact]{titlesec}
\usepackage{titlesec}

\renewcommand{\thesection}{\arabic{section}}

\usepackage{float}
\usepackage[most]{tcolorbox}
\usepackage{hyperref}
 \hypersetup{
     colorlinks=true,
     linkcolor=Navy,
     filecolor=blue,
     citecolor = black,      
     urlcolor=cyan,
     }

\setlength\parindent{0pt} % Gets rid of all indentation


\pagestyle{fancy} % Enables the custom headers/footers
% \usepackage[compact]{titlesec}

\lhead{} \rhead{} % Headers - all  empty

\lhead{\color{Grey}Estelle Long-Merle}  % Write your name here
\rhead{\color{Grey}L3S2 MIASHS - Introduction à l'apprentissage automatique}
\cfoot{\color{Grey} \thepage}

\renewcommand{\headrulewidth}{0.4pt} % No header rule
\renewcommand{\footrulewidth}{0pt} % Thin footer rule
% \usepackage{enotez}
% \setenotez{list-name=Notes}

% \let\footnote=\endnote


% \title{}
% \author{}
% \date{}

\begin{document}

\begin{titlepage}
   \begin{center}
       \vspace*{1cm}
                   
       \textbf{L3S2 MIASHS - Introduction à l'apprentissage automatique}
       
       \vfill

       \textbf{\color{DarkRed} \Huge Application d'algorithmes de classification supervisée pour prédire la satisfaction de passagers d'une compagnie aérienne}

       \vspace{0.5cm}
        
    \textbf{\Large Dossier de validation}
    
       \vspace{3cm}

       \textbf{\Large Estelle Long-Merle | No étudiant: 12106173}

       \vfill
            
       \vspace{0.8cm}
     
       \includegraphics[width=0.4\textwidth]{pictures/logo-uga.png}
            
       Université Grenoble Alpes\\
       Licence 3 MIASHS
            
   \end{center}
\end{titlepage}

\tableofcontents
\newpage

\section{Description du jeu de données et présentation de la problématique}

Le jeu de données utilisé est``airline passenger satisfaction'' qui décrit la satisfaction des passagers à bord d'une compagnie aérienne anonyme. Il s'agit probablement de la compagnie aérienne United Airlines au vue de la classe Economy Plus. Il vise à comprendre les
facteurs qui influencent la satisfaction des passagers lors de leurs
voyages en avion.
\vspace{0.5cm}
\\
Le jeu de données est disponible sur Kaggle :

\href{https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction}{\uline{https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction}}
\vspace{0.5cm}
\newline
Le code utilisé pour le projet est disponible au lien github suivant :

\href{https://github.com/estellelm38/airline_satisfaction}{\uline{https://github.com/estellelm38/airline\_satisfaction}}

\section{Data preprocessing}
Nous cherchons à comprendre quels facteurs influencent la satisfaction des passagers aériens et quelle est leur influence respective.
L'ensemble de données comprend un total de 129 880 observations et 22 variables, parmi lesquelles figurent des variables catégorielles, quantitatives et ordinales.
La target est le niveau de satisfaction du client de la compagnie aérienne dont les modalités sont : satisfied / neutral or dissatisfied.
Le jeu de données est équilibré.

\vspace{0.5cm}
Nous cherchons à obtenir la meilleure accuracy et la meilleure précision sur la target.

Les variables catégorielles comprennent des informations telles que le genre, le type de client, le type de voyage et la classe de vol. Les variables quantitatives comprennent l'âge, la distance de vol, ainsi que les retards au départ et à l'arrivée. Les variables ordinales représentent la satisfaction des passagers à l'égard de différents services à bord, tels que le wifi à bord, la commodité des horaires de départ/arrivée, la réservation en ligne, etc. (\hyperref[annexe:Table1]{table 1})
\vspace{0.5cm}

Nous commençons par observer les distributions des différentes variables à l'aide d'histogrammes de countplots et de diagrammes pour mieux visualiser le jeu de données, les graphiques obtenus sont disponibles en annexe. (\hyperref[annexe:Table1]{table 2})

\subsection{Encodage}

Les modèles étant basés sur des calculs mathématiques, il est nécessaire de représenter les données catégorielles et ordinales numériquement à l'aide de techniques d'encodage afin qu'elles soient comprises par les modèles.

Nous cherchons à encoder des features, Nous allons donc utiliser les deux encodeurs proposés par scikit learn dans ce cas de figure, à savoir one hot encoder et ordinal encoder, afin d'encoder respectivement les variables catégorielles et ordinales.
\vspace{0.5cm} \\
Nous utilisons one hot encoder pour l'encodage des variables catégorielles afin d'éviter de créer des relations d'ordre entre ces variables et pénaliser les modèles sensibles.
Nous allons ici décomposer les variables en sous variables et créer autant de colonnes que nous avons de catégories (celles-ci sont représentées de façon binaires). Nous obtenons une sparse matrix très large qui est automatiquement compressée. 
Après encodage, 6 nouvelles colonnes ont été générées (4 pour les features et 2 pour la target). (\hyperref[annexe:image1]{figure 1})
\vspace{0.5cm} \\
Nous utilisons l’ordinal encoder pour l'encodage des variables ordinales
Il permet d'associer à chaque classe une valeur numérique en traitant une par une les différentes colonnes. Cela a permis de représenter les différentes catégories des variables ordinales par des entiers, tout en préservant l'ordre des catégories. (\hyperref[annexe:image2]{figure 2})

\subsection{Mise à l'échelle}

Il faut mettre toutes les features à la même échelle car par exemple l'âge est de l’ordre de la dizaine et la distance est de l’ordre du millier et certains classifieurs fonctionnent selon des distances. (\hyperref[annexe:image3]{figure 3})
\vspace{0.5cm} \\
Le MinMax scaling permet de mettre à l'échelle chaque feature pour que ses valeurs soient comprises entre 0 et 1. La formule suivante est appliquée à toutes les valeurs de chaque feature: : 

\[
\frac{{X - X_{\min}}}{{X_{\max} - X_{\min}}}
\]
\\
avec $X$  la valeur d'entrée, $X_{\min}$ la valeur minimum et $X_{\max}$ la valeur maximum de la feature.
\\
Ce processus permet de comparer les différentes features sans perdre d’information car les distances entre leurs valeurs restent proportionnelles entre elles. (\hyperref[annexe:image4]{figure 4})
\vspace{0.5cm} \\
Nous avons également testé standard scaling et robust scaling cependant ceux-ci ont donné de moins bon résultats ce qui peut être expliqué pa la grande variance des données. Pour standard scaling afin de donner un écart type de 1, celui-ci a produit plusieurs valeurs négatives qui ont par la suite complexifié les calculs de certains modèles d'apprentissage. De plus, les valeurs aberrantes sont rares mais importants pour le jeu de données comme on le verra juste après. Ainsi il est moins intéressant d'utiliser le robust scaling qui perd de l'information en utilisant un calcul de médiane plutôt qu'une moyenne au profit d'une robustesse face aux valeurs aberrantes. On va ici privilégier au contraire une méthode sensible à ces valeurs aberrantes.

\subsection{Valeurs aberrantes}

Concernant les valeurs aberrantes, nous avons choisi de les garder dans le jeu de données.
En effet, en ce qui concerne les délais de départ/arrivée nous avons jugé qu’une valeur aberrante aurait une influence sur la satisfaction du passager et donc il est important de ne pas la supprimer. Ensuite nous ne supprimons pas les valeurs aberrantes des distances de vol car il s'agit de valeurs plausibles (la distance maximum dans le dataset : 4983 km, est inférieure à la distance d’un vol Russie-USA : 8881 km)

\subsection{Sélection des features}

Nous devons sélectionner les variables les plus utiles pour nos modèles afin d'améliorer leurs performances.

Nous avons tracé la matrice des corrélation (\hyperref[annexe:image5]{figure 5}) puis avons affiché les features qui ont une corrélation |r| > 0.8. (\hyperref[annexe:image6]{figure 6})\\
Nous avons ensuite supprimé une feature pour chaque paire de features corrélées sur le critère de garder une bonne compréhension, par exemple en gardant seulement ‘satisfied’ et en supprimant ‘unsatisfied’.

Nous avons supprimé la feature délai de départ en minutes car il 'agissait de la seule qui comportait des valeurs manquantes et ce en grande quantité (393) (\hyperref[annexe:image7]{figure 7}). Ainsi nous avons préféré la supprimer et garder la feature délai de départ à l'arrivée pour éviter d'avoir à imputer les valeurs manquantes. 
\\

Nous avons ensuite tracé deux graphiques : l'un représentant l'importance de chaque feature (\hyperref[annexe:image8]{figure 8}) et l'autre l'importance des features cumulée (\hyperref[annexe:image9]{figure 9}) qui dépend du modèle de forêt aléatoires. Néanmoins, nous avons obtenu de bons résultats en gardant toutes les features.

\section{Apprentissage supervisé}

Pour la phase d'apprentissage supervisé, nous avons testé les 4 classifieurs suivants : Le K-Nearest Neighbors (KNN), le Naive Bayes, des forêts aléatoires et le SVC ainsi que la classification par un réseau de neurones (MLP).
\vspace{0.5cm} \\
On a décidé de représenter la qualité de l’apprentissage pour chaque modèle à l’aide d’un graphique ROC c’est une mesure de la performance pour les classificateurs binaires qui donne le taux de vrais positifs en fonction du taux de faux positifs
\subsection{K plus proches voisins}

L'algorithme des k plus proches voisins (KNN) est une méthode d'apprentissage supervisé utilisée pour la classification et la régression. Son fonctionnement repose sur le principe suivant : les instances similaires se regroupent dans l'espace des caractéristiques.
\vspace{0.5cm} \\
Dans notre étude, nous avons utilisé l'estimateur KNeighborsClassifier pour implémenter KNN. Notre première étape a été de diviser notre ensemble de données en un ensemble d'entraînement (80 pourcents des données) et un ensemble de test (20 pourcents des données), une pratique courante en apprentissage automatique.
\vspace{0.5cm} \\
L'hyperparamètre principal de KNN est le nombre de voisins (k), qui détermine le nombre d'instances les plus proches à considérer lors de la prédiction. De plus, il est nécessaire de choisir une méthode de calcul de la distance, comme la distance euclidienne, la distance de Manhattan ou la distance de Minkowski.
\vspace{0.5cm} \\
Pour évaluer la performance de notre modèle, nous avons utilisé la validation croisée. Nous avons opté pour une validation K-fold, qui consiste à diviser l'ensemble de données en k sous-ensembles et à évaluer le modèle k fois en utilisant chaque sous-ensemble comme ensemble de test une fois et les autres comme ensemble d'entraînement. Cela nous a permis d'obtenir une estimation plus fiable des performances du modèle en calculant la moyenne des mesures de performance sur chaque pli.
\vspace{0.5cm} \\
En répétant ce processus pour différentes valeurs de k (de k=1 à k=10 dans notre cas), nous avons pu déterminer l'hyperparamètre optimal (k=7 dans notre cas) qui maximise les performances du modèle. Cette approche nous a également permis d'identifier les splits de données les plus appropriés pour l'entraînement et la validation du modèle.
\vspace{0.5cm} \\
Nous avons obtenu une accuracy moyenne de 0.93 (\hyperref[annexe:image10]{figure 10}) et une précision moyenne de 0.97 (\hyperref[annexe:image11]{figure 11}) lors de la validation K-fold pour notre modèle KNN et avons représenté l'efficacité de ce modèle à l'aide d'un graphique ROC. 
(\hyperref[annexe:image12]{figure 12}) 
\vspace{0.5cm} \\
Bien que le classifieur KNN soit relativement simple à comprendre, il présente des inconvénients tels que sa sensibilité au bruit et sa lenteur en raison du calcul de toutes les distances pour chaque instance. Cependant, les résultats obtenus fournissent une base solide pour l'analyse et peuvent être utilisés comme point de départ pour des études plus approfondies.

\subsection{Classifieur bayésien naïf}

Les classifieurs bayésiens naïfs sont des modèles probabilistes utilisés pour estimer la probabilité d'appartenance à une classe donnée étant donné un vecteur de distribution. Ils reposent sur le théorème de Bayes pour calculer cette probabilité en se basant sur les distributions conditionnelles des caractéristiques (features) et la probabilité a priori des classes.
\vspace{0.5cm} 
Ces classifieurs probabilistes font l'hypothèse que les caractéristiques sont indépendantes les unes des autres, ce qui simplifie le calcul des probabilités conditionnelles. Ils n'ont pas d'hyperparamètres à régler et estiment les paramètres des distributions en découpant le jeu de données en fonction des classes cibles, puis en calculant les moyennes et les écarts-types (ou d'autres paramètres selon la distribution choisie) pour chaque classe.
\vspace{0.5cm} \\
Dans notre étude, nous avons utilisé les estimateurs Gaussian Naive Bayes et Categorical Naive Bayes pour traiter à la fois les variables quantitatives et catégorielles. Après avoir divisé l'ensemble de données en ensembles d'entraînement et de test, puis avoir effectué une validation croisée, nous avons obtenu une accuracy moyenne d'environ 0.68 pour les variables quantitatives et d'environ 0.89 pour les variables catégorielles. Sur l'ensemble du jeu de données, nous avons obtenu une accuracy moyenne de 0.82 et une précision moyenne de 0.82. (\hyperref[annexe:image13]{figure 13}) Nous avons représenté l'efficacité de ce modèle à l'aide d'un graphique ROC. (\hyperref[annexe:image14]{figure 14})
\vspace{0.5cm} \\
Les avantages des classifieurs bayésiens naïfs résident dans leur capacité à prendre en compte des connaissances externes pour estimer les distributions, ainsi que dans leur rapidité de calcul. Cependant, ils présentent des limites en raison de leurs suppositions de distribution et d'indépendance conditionnelle. Bien que cet algorithme soit rapide et efficace avec un grand nombre de données, il ne prend pas en compte les liens entre certaines caractéristiques et est connu pour être un mauvais estimateur de probabilité, ce qui limite son utilisation dans certaines situations.

\subsection{Forêts aléatoires}

Les forêts aléatoires sont des modèles d'apprentissage automatique qui combinent plusieurs arbres de décision pour améliorer la précision de la prédiction et réduire le surapprentissage.
\vspace{0.5cm}
Au départ, plusieurs arbres de décision sont entraînés sur des sous-ensembles aléatoires du jeu de données d'entraînement. Chaque arbre est formé sur un échantillon de données tiré avec remplacement (bootstrap) et sur un sous-ensemble aléatoire des caractéristiques (features).
\vspace{0.5cm} \\
Les hyperparamètres à tester comprennent le nombre d'arbres de décision à inclure dans la forêt, la profondeur maximale des arbres pour éviter le surapprentissage, et le nombre minimum d'échantillons requis pour diviser un nœud en deux.
\vspace{0.5cm} \\
Chaque arbre de décision est construit en choisissant à chaque étape le meilleur split (avec le critère de Gini), parmi un sous-ensemble aléatoire des caractéristiques. Cela continue jusqu'à ce que les feuilles des arbres soient "pures" ou que la profondeur maximale de l'arbre soit atteinte.
\vspace{0.5cm} \\
Pour effectuer une prédiction, chaque arbre de la forêt donne une réponse basée sur les caractéristiques d'entrée. Dans le cas d'une classification, le résultat final est déterminé par un vote majoritaire parmi tous les arbres.
\vspace{0.5cm} \\
En combinant plusieurs arbres formés sur des sous-ensembles de données différents, les forêts aléatoires réduisent le surapprentissage par rapport à un seul arbre de décision et augmentent la précision. Cela les rend plus robustes et moins sensibles aux variations du jeu de données d'entraînement.
\vspace{0.5cm} \\
Les forêts aléatoires fournissent également une mesure de l'importance de chaque caractéristique dans la prédiction. Cette importance est calculée en mesurant la réduction moyenne de l'impureté (par exemple, l'indice de Gini) que chaque caractéristique apporte lorsqu'elle est utilisée pour diviser les arbres de la forêt.
\vspace{0.5cm} \\
Nous avons utilisé l'estimateur RandomForestClassifier et effectué une KFold validation. Nous avons obtenu une accuracy moyenne de 0.96 (\hyperref[annexe:image15]{figure 15}) sur l'ensemble des données avec 100 estimateurs (arbres).
\vspace{0.5cm} \\
Les forêts aléatoires sont robustes, peuvent gérer des données manquantes et des caractéristiques catégorielles sans prétraitement supplémentaire, et sont moins sensibles au surapprentissage par rapport aux arbres de décision simples.
\vspace{0.5cm} \\
Dans l’ensemble, les résultats montrent que le classifieur le plus intéressant pour notre jeu de données en apprentissage supervisé semble être celui des forêts aléatoires.

\subsection{SVM à noyeau}
\subsection{Réseau de neurones (MLP)}

\section{PCA}
\subsection{K plus proches voisins avec PCA}
\subsection{Classifieur bayésien naïf avec PCA}
\subsection{Forêts aléatoires avec PCA}

\section{Apprentissage non-supervisé}
\subsection{CAH}

\section{Résumé et perspectives}

\clearpage

\section{Annexe}

\begin{table} [htbp]
    \centering
    \small
    \center{\caption{features du dataset airline passenger satisfaction triées par type} }
    \begin{tabular}{ccc}
        \toprule
        Quantitatives & Catégorielles & Ordinales\\
        \midrule
        Age & Gender (Male/Female) & Inflight wifi service \\
        Flight Distance & Customer Type (Loyal/Disloyal) & Departure/Arrival time convenient \\
        Departure Delay in Minutes & Type of Travel (Personal/Business) & Ease of Online booking \\
        Arrival Delay in Minutes & Class (Eco plus/Business/Eco) & Gate location \\
        & & Food and drink \\
        & & Online boarding \\
        & & Seat comfort \\
        & & Inflight entertainment \\
        & & On-board service \\
        & & Leg room service \\
        & & Baggage handling \\
        & & Checkin service \\
        & & Inflight service \\
        & & Cleanliness \\
        \bottomrule
    \label{annexe:Table1}
    \end{tabular}
\end{table}

\begin{table} [htbp]
    \centering
    \caption{répartition des individus par features dans le dataset}
    \begin{tabular}{ccc}
        \includegraphics[width=0.3\linewidth]{pictures/image19.png} &
        \includegraphics[width=0.3\linewidth]{pictures/image20.png} &
        \includegraphics[width=0.3\linewidth]{pictures/image23.png} \\
        \includegraphics[width=0.3\linewidth]{pictures/image24.png} &
        \includegraphics[width=0.3\linewidth]{pictures/image25.png} &
        \includegraphics[width=0.3\linewidth]{pictures/image26.png} \\
        \includegraphics[width=0.3\linewidth]{pictures/image27.png} &
        \includegraphics[width=0.3\linewidth]{pictures/image28.png} &
        \includegraphics[width=0.3\linewidth]{pictures/image29.png} \\
        \includegraphics[width=0.3\linewidth]{pictures/image30.png} &
        \includegraphics[width=0.3\linewidth]{pictures/image31.png} &
        \includegraphics[width=0.3\linewidth]{pictures/image32.png} \\
        \includegraphics[width=0.3\linewidth]{pictures/image33.png} &
        \includegraphics[width=0.3\linewidth]{pictures/image34.png} &
        \includegraphics[width=0.3\linewidth]{pictures/image35.png} \\
        \includegraphics[width=0.3\linewidth]{pictures/image36.png} &
        \includegraphics[width=0.3\linewidth]{pictures/image37.png} &
        \includegraphics[width=0.3\linewidth]{pictures/image38.png} \\
        \includegraphics[width=0.3\linewidth]{pictures/image39.png} &
        \includegraphics[width=0.3\linewidth]{pictures/image40.png} &
        \includegraphics[width=0.3\linewidth]{pictures/image21.png} \\
    \label{annexe:Table2}
    \end{tabular}
\end{table}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image1.png}
    \caption{variables catégorielles une fois encodée avec le one-hot encode}
    \label{annexe:image1}
\end{figure}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image2.png}
    \caption{variables ordinales une fois encodée avec l’ordinal encoder}
    \label{annexe:image2}
\end{figure}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image41.png}
    \caption{variables ordinales une fois encodée avec l’ordinal encoder}
    \label{annexe:image3}
\end{figure}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image3.png}
    \caption{variables ordinales une fois encodée avec l’ordinal encoder}
    \label{annexe:image4}
\end{figure}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image52.png}
    \caption{variables ordinales une fois encodée avec l’ordinal encoder}
    \label{annexe:image5}
\end{figure}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image4.png}
    \caption{variables ordinales une fois encodée avec l’ordinal encoder}
    \label{annexe:image6}
\end{figure}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image5.png}
    \caption{variables ordinales une fois encodée avec l’ordinal encoder}
    \label{annexe:image7}
\end{figure}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image42.png}
    \caption{variables ordinales une fois encodée avec l’ordinal encoder}
    \label{annexe:image8}
\end{figure}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image43.png}
    \caption{variables ordinales une fois encodée avec l’ordinal encoder}
    \label{annexe:image9}
\end{figure}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image6.png}
    \caption{variables ordinales une fois encodée avec l’ordinal encoder}
    \label{annexe:image10}
\end{figure}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image7.png}
    \caption{variables ordinales une fois encodée avec l’ordinal encoder}
    \label{annexe:image11}
\end{figure}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image44.png}
    \caption{variables ordinales une fois encodée avec l’ordinal encoder}
    \label{annexe:image12}
\end{figure}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image8.png}
    \caption{variables ordinales une fois encodée avec l’ordinal encoder}
    \label{annexe:image13}
\end{figure}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image45.png}
    \caption{variables ordinales une fois encodée avec l’ordinal encoder}
    \label{annexe:image14}
\end{figure}

\begin{figure} [htbp]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image9.png}
    \caption{variables ordinales une fois encodée avec l’ordinal encoder}
    \label{annexe:image15}
\end{figure}

\printbibliography

\end{document}
