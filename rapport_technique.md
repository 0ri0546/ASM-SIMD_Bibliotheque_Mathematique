# Rapport technique — Bibliothèque mathématique SIMD

## 1. Introduction et état initial

### 1.1 Objectif du projet

L'objectif de ce projet est d'optimiser une bibliothèque mathématique C++ en exploitant les instructions SIMD (*Single Instruction, Multiple Data*). Cette approche permet d'appliquer une même opération à plusieurs valeurs simultanément, notamment grâce aux registres vectoriels du processeur.

Le projet consiste à conserver une implémentation C++ de référence, à développer une version optimisée utilisant les instructions SIMD et à comparer leurs performances dans des conditions reproductibles.

Le périmètre comprend les vecteurs, les matrices, les quaternions ainsi que trois traitements par lots : le calcul de produits scalaires, la normalisation de vecteurs et la transformation de points 3D.

### 1.2 Bibliothèque de référence

Le projet s'appuie sur une bibliothèque mathématique préexistante réalisée par Dylan, issue de son travail sur NexusEngine et inspirée d'un projet de bibliothèque mathématique développé durant la deuxième année.

Sources :
- [Bibliothèque NexusEngine](<https://github.com/dark-dylan-dev/NexusEngine/tree/d6f8c309c47465c5be1c3de36379b47e6365c0b1/Source/Engine/Core/Math/Public>)
- [Projet de deuxième année](<https://github.com/GamingCampus-MillieBourgois-25-26/3d-geometry-hello-kitty/tree/master/Library>)

La bibliothèque a été reprise, corrigée et complétée pour répondre aux besoins du projet. Une adaptation de l'organisation du code a notamment été réalisée, avec le passage des modules aux fichiers d'en-tête.

Le socle de référence prend en charge les types `float` et `double`, les vecteurs 3D, les matrices 4 × 4, les quaternions et les opérations mathématiques associées.

Cette version de référence est conservée afin de servir de base aux tests de validation et aux comparaisons de performances.

## 2. Architecture et implémentation

### 2.1 Organisation de la solution

La solution Visual Studio est développée en C++23 pour Windows x64. Elle sépare les éléments suivants :

- La bibliothèque mathématique de référence, utilisée comme base de comparaison.
- La bibliothèque SIMD, contenant les implémentations optimisées.
- Le projet de tests unitaires, chargé de vérifier la validité des opérations.
- Le projet de benchmark, utilisé pour mesurer et comparer les performances.

Cette séparation permet de comparer les implémentations sans supprimer la version initiale et de distinguer la validation fonctionnelle de l'évaluation des performances.

### 2.2 Types et instructions SIMD

Les implémentations couvrent les types `float` et `double`. Les optimisations s'appuient sur des instructions SSE/SSE2 et AVX2, selon les opérations et les types traités.

Les largeurs SIMD prises en charge sont de 128 et 256 bits. La largeur effective dépend du type numérique : un registre de 128 bits contient quatre valeurs `float` ou deux valeurs `double`, tandis qu'un registre de 256 bits en contient respectivement huit ou quatre.

Les instructions SIMD permettent de traiter plusieurs valeurs au cours d'une même opération. Cependant, leur utilisation ne garantit pas automatiquement une amélioration des performances : le coût de préparation des données, leur disposition en mémoire et les opérations nécessaires peuvent limiter le gain obtenu.

### 2.3 Conventions mathématiques

Les conventions de stockage des matrices, d'indexation des éléments et de composition des transformations sont documentées dans le code. Cette clarification est importante pour garantir que les implémentations de référence et SIMD effectuent les mêmes calculs.

Le comportement de la normalisation d'un vecteur nul est également défini afin d'éviter une division par zéro et de conserver un comportement cohérent entre les différentes implémentations.

### 2.4 Fonction assembleur x64

Une fonction assembleur x64, nommée `Vec4fAdd`, est appelée depuis le code C++. Elle réalise une addition de vecteurs de quatre valeurs flottantes en modifiant le premier opérande.

Cette fonction permet d'étudier l'intégration d'un noyau assembleur dans un projet C++ et de comprendre le fonctionnement des opérations vectorielles à bas niveau.

Deux extraits de désassemblage commentés complètent cette partie :
- `SIMD/asm_comments_Vec4f_operatorPlusEqual.txt`
- `SIMD/asm_comments_Vec4f_normalized.txt`

Ces extraits permettent d'observer les instructions générées pour deux opérations et de relier le code de haut niveau aux instructions exécutées par le processeur.

## 3. Traitements par lots et organisation mémoire

### 3.1 Traitements étudiés

Trois traitements par lots ont été implémentés en version de référence et en version SIMD, en `float` et en `double`.

**Produit scalaire de deux tableaux de vecteurs**

Pour chaque indice, le produit scalaire de deux vecteurs 3D est calculé :

```latex
\[
a \cdot b = a_xb_x + a_yb_y + a_zb_z
\]
```

Le traitement produit un résultat scalaire pour chaque paire de vecteurs.

**Normalisation de vecteurs**

Chaque vecteur est divisé par sa longueur afin d'obtenir un vecteur unitaire :

```latex
\[
\hat{v} = \frac{v}{\sqrt{v_x^2+v_y^2+v_z^2}}
\]
```

Le cas du vecteur nul est traité séparément pour éviter une division par zéro. Le comportement défini pour ce cas est conservé entre les implémentations.

**Transformation de points 3D**

Chaque point 3D est transformé par une même matrice affine 4 × 4. La matrice est réutilisée pour l'ensemble du lot, ce qui permet d'étudier le traitement de nombreuses données avec une transformation commune.

### 3.2 Gestion des tailles de lots

Les traitements sont conçus pour accepter des lots vides, de petite taille ou dont le nombre d'éléments n'est pas un multiple de la largeur SIMD.

Les éléments pouvant être traités par registres vectoriels sont calculés par groupes. Les éléments restants sont ensuite traités sans lecture ni écriture hors limites.

Cette gestion est essentielle pour assurer la correction des traitements, quelle que soit la taille du lot.

### 3.3 Comparaison AoS et SoA

Deux organisations mémoire sont comparées.

- **AoS (*Array of Structures*)** : les composantes d'un vecteur sont stockées ensemble, puis les vecteurs sont placés les uns à la suite des autres.
- **SoA (*Structure of Arrays*)** : les composantes sont réparties dans des tableaux distincts, par exemple un tableau pour les coordonnées X, un pour Y et un pour Z.

AoS correspond à une organisation naturelle pour manipuler individuellement des objets vectoriels. SoA peut faciliter les chargements contigus de données et l'application d'une même opération à plusieurs valeurs d'une composante.

La comparaison est effectuée pour les trois traitements, en `float` et en `double`, avec les versions de référence et SIMD. Les résultats AoS/SoA sont présentés séparément des comparaisons référence/SIMD afin de distinguer les deux effets.

## 4. Protocole expérimental

### 4.1 Environnement

Les mesures ont été effectuées sur la même machine, sous Windows x64, avec une compilation Release x64.

| Paramètre | Valeur |
|---|---|
| Processeur | AMD Ryzen 7 5700G with Radeon Graphics |
| Environnement | Ordinateur du Gaming Campus, salle Eliott, 018 |
| Compilation | Release x64 |
| Langage | C++23 |
| ISA indiquée par le benchmark | AVX2 |
| Tailles de lots | 1 000, 100 000 et 1 000 000 éléments |
| Itérations | 10 |
| Échantillons | 10 |
| Échauffement | 10 ms |
| Seed | 305419896 |

La graine aléatoire est fixée afin de rendre les données d'entrée reproductibles. La configuration d'exécution est récupérée et affichée par le benchmark.

### 4.2 Méthode de mesure

Les données sont préparées avant le début du chronométrage afin de limiter l'influence de leur initialisation sur le temps mesuré. Les allocations nécessaires aux données de test sont également effectuées en dehors de la zone chronométrée lorsque le scénario le permet.

Une période d'échauffement de 10 ms est utilisée avant les mesures. Le benchmark utilise une fonction `DoNotOptimizeAway` pour limiter le risque que le compilateur supprime les calculs dont les résultats ne seraient pas utilisés.

Les mesures sont répétées sur plusieurs échantillons. Le programme présente notamment la médiane, la dispersion et le rapport entre le temps de référence et le temps SIMD.

Le *speedup* est calculé ainsi :

```latex
\[
\text{Speedup} =
\frac{\text{Temps de référence}}
{\text{Temps SIMD}}
\]
```

Un speedup supérieur à 1 indique que la version SIMD est plus rapide pour le scénario mesuré. Une valeur inférieure à 1 signifie qu'elle est plus lente. Une valeur proche de 1 indique des performances similaires.

### 4.3 Limites du protocole

La comparaison principale oppose le C++ optimisé à la version SIMD explicite. Le compilateur peut déjà vectoriser automatiquement certaines opérations dans le code C++ de référence. Il ne s'agit donc pas nécessairement d'une comparaison entre du code scalaire pur et du code vectorisé.

Les résultats dépendent également du processeur, du compilateur, des options de compilation, de la disposition mémoire et de la taille des lots. Les conclusions doivent être interprétées dans le contexte de la configuration testée.

## 5. Résultats et analyse des benchmarks

Les résultats bruts sont conservés dans `Benchmark/Benchmark.csv`. Le benchmark produit des comparaisons entre les implémentations de référence et SIMD ainsi qu'une analyse distincte de l'organisation mémoire AoS/SoA.

### 5.1 Comparaison référence/SIMD

Les mesures montrent que l'utilisation explicite de SIMD n'améliore pas systématiquement les performances. Le gain dépend notamment du traitement étudié et de l'organisation des données.

Dans les résultats enregistrés, les traitements SoA présentent des gains particulièrement importants pour certaines opérations. À l'inverse, plusieurs traitements AoS sont plus lents dans leur version SIMD explicite que dans leur version de référence optimisée.

Les résultats disponibles comprennent notamment les observations suivantes pour les traitements par lots :

| Traitement SIMD | Speedup observé | Interprétation |
|---|---:|---|
| Produit scalaire `float`, AoS | 0,298× | SIMD plus lent |
| Produit scalaire `double`, AoS | 0,540× | SIMD plus lent |
| Normalisation `float`, AoS | 0,775× | SIMD plus lent |
| Normalisation `double`, AoS | 0,836× | SIMD plus lent |
| Normalisation `float`, SoA | 2,563× | SIMD plus rapide |
| Normalisation `double`, SoA | 1,716× | SIMD plus rapide |
| Transformation de points `float`, AoS | 0,589× | SIMD plus lent |
| Transformation de points `double`, AoS | 0,844× | SIMD plus lent |
| Transformation de points `float`, SoA | 3,824× | SIMD plus rapide |
| Transformation de points `double`, SoA | 1,187× | SIMD plus rapide |

Ces valeurs illustrent les résultats du benchmark enregistré ; elles doivent être associées à la configuration et à la taille de lot correspondantes. Le fichier CSV constitue la source de référence pour vérifier les mesures finales.

Les résultats suggèrent que les traitements SoA tirent davantage parti de l'exécution vectorielle explicite dans plusieurs scénarios. À l'inverse, les traitements AoS peuvent nécessiter des chargements, des extractions ou des rassemblements de composantes qui réduisent l'avantage des instructions SIMD.

### 5.2 Comparaison AoS/SoA

La comparaison AoS/SoA mesure l'influence de la disposition des données indépendamment de la comparaison entre référence et SIMD.

Les résultats de référence indiquent que SoA est plus rapide pour plusieurs traitements, notamment le produit scalaire et la normalisation en `float`. Toutefois, ce comportement n'est pas universel : la transformation de points en `float` constitue un cas où AoS est plus rapide dans les mesures enregistrées.

Pour les implémentations SIMD, SoA présente des avantages marqués dans les scénarios observés. Dans les résultats enregistrés, les gains SoA atteignent notamment environ 4,6× pour le produit scalaire en `float`, 4,7× pour la normalisation en `float` et 4,6× pour la transformation de points en `float`.

Ces observations sont cohérentes avec le fait que SoA permet de charger directement plusieurs valeurs d'une même composante dans un registre vectoriel. En AoS, les composantes de différents vecteurs sont entrelacées en mémoire ; leur traitement simultané peut nécessiter davantage de manipulations.

### 5.3 Interprétation des résultats

Les mesures montrent que le choix de l'organisation mémoire est aussi important que le choix des instructions SIMD. Une implémentation vectorisée peut être moins performante si elle doit consacrer une part importante du travail à organiser les données avant les calculs.

De plus, le code C++ de référence étant compilé en mode Release, le compilateur peut appliquer ses propres optimisations, notamment l'auto-vectorisation. L'écart mesuré correspond donc au bénéfice réel de l'implémentation SIMD explicite par rapport au code de référence optimisé.

Il serait incorrect de conclure qu'une approche est toujours supérieure à l'autre à partir d'un seul résultat. Les speedups doivent être examinés opération par opération, pour chaque type numérique, organisation mémoire et taille de lot.

## 6. Analyse bas niveau et profiling CPU

### 6.1 Profiling avec Visual Studio

Une session de profiling CPU a été réalisée avec Visual Studio Performance Profiler. Le fichier de session est conservé dans `Benchmark/Rapport20261008-1010.diagsession`, et son commentaire détaillé dans `Benchmark/CPU_Profiling_Commentary.txt`.

Dans la vue de l'arbre des appels, `BenchmarkMatSIMD` représente 29,50 % du temps processeur total observé, contre 19,91 % pour `BenchmarkMatNoSIMD`. Ces pourcentages correspondent à la part de temps attribuée aux fonctions pendant cette session ; ils ne constituent pas une comparaison directe de leur vitesse.

Le temps CPU exclusif de `BenchmarkMatSIMD` est de 0,22 %. La majeure partie du temps attribué à cette fonction provient donc des fonctions appelées. L'arbre des appels permet d'examiner ces branches, mais les captures ne suffisent pas à attribuer l'intégralité du coût à une opération mathématique particulière.

La présence de fonctions telles que `free` ou `std::filesystem::filesystem_error::_Pretty_message` ne prouve pas à elle seule qu'une erreur a été déclenchée ni qu'elles constituent un goulot d'étranglement. Une analyse complémentaire serait nécessaire pour établir leur origine et leur impact éventuel.

Le profiling complète les benchmarks en aidant à identifier les chemins d'exécution coûteux. Pour comparer la vitesse des implémentations, les mesures de temps par opération restent plus appropriées que les pourcentages du temps total de profiling.

### 6.2 Désassemblage

Deux extraits de désassemblage ont été commentés : l'un associé à une opération d'addition de vecteurs et l'autre à la normalisation.

Cette analyse permet de relier les opérations du code C++ aux instructions de bas niveau et de comprendre comment certaines opérations sont réalisées par le processeur.

Le code assembleur `Vec4fAdd` constitue également un exemple d'intégration d'une fonction x64 dans la bibliothèque. L'utilisation de ces instructions permet d'étudier concrètement la vectorisation, mais ne garantit pas à elle seule une accélération : celle-ci doit être confirmée par des mesures effectuées dans des conditions comparables.

## 7. Validation, limites et conclusion

### 7.1 Validation fonctionnelle

La suite de tests automatisés a été exécutée, avec un total annoncé de 573 tests réalisés. Les tests servent à vérifier la cohérence des opérations et à détecter les régressions entre les différentes implémentations.

Les cas étudiés comprennent les opérations mathématiques courantes, les traitements par lots, les tailles non multiples de la largeur SIMD et la gestion du vecteur nul. Les traitements doivent également fonctionner avec des lots vides et sans accès mémoire hors limites.

Les écarts numériques entre les versions doivent être interprétés en tenant compte des différences possibles d'ordre d'exécution des opérations flottantes. Pour finaliser cette section, les tolérances employées par les tests et, si elles sont disponibles, les erreurs numériques maximales mesurées doivent être indiquées.

### 7.2 Limites

Les résultats dépendent de la machine et des options de compilation utilisées. Ils ne peuvent pas être généralisés à tous les processeurs sans mesures complémentaires.

Les instructions SIMD explicites peuvent ne pas être avantageuses lorsque l'organisation mémoire entraîne des opérations supplémentaires ou lorsque le compilateur optimise déjà efficacement la version de référence. Les résultats AoS/SoA montrent également que la meilleure organisation dépend du traitement.

Le projet étudie principalement les performances sur une configuration Windows x64 et dans les conditions du protocole décrit. Des essais supplémentaires sur d'autres processeurs, avec d'autres tailles de lots et une analyse plus détaillée des instructions générées permettraient d'approfondir les conclusions.

### 7.3 Conclusion

Ce projet a permis de conserver une bibliothèque mathématique de référence, de développer des implémentations SIMD et de mettre en place un protocole de comparaison reproductible.

Les benchmarks montrent que les gains dépendent fortement du traitement et de l'organisation mémoire. Les implémentations SoA SIMD présentent des améliorations importantes pour plusieurs opérations, tandis que certaines implémentations AoS SIMD restent moins performantes que le C++ de référence optimisé.

La validation fonctionnelle, les benchmarks, le profiling CPU et l'analyse du désassemblage apportent des informations complémentaires : les tests vérifient la correction, les benchmarks mesurent les performances et les outils d'analyse aident à comprendre le comportement du programme.

L'enseignement principal est que la vectorisation ne suffit pas, à elle seule, à garantir une accélération. Pour obtenir de bonnes performances, il faut également tenir compte de la disposition des données, du coût des chargements et manipulations, ainsi que des optimisations déjà effectuées par le compilateur.

### 7.4 Utilisation de l'intelligence artificielle

L'intelligence artificielle a été utilisée pour certaines tâches de rédaction répétitives, notamment la rédaction de tests unitaires, la documentation de registres et la traduction de ce rapport. L'analyse des résultats, l'architecture du projet et les calculs ont été réalisés sans IA.

## 8. Livrables

Les livrables associés au projet sont :

- Le dépôt Git individuel, comprenant l'historique des modifications, la solution Visual Studio et les fichiers nécessaires à la compilation.
- La bibliothèque mathématique, avec ses implémentations de référence et SIMD.
- L'application console de benchmark en configuration Release x64.
- La fonction assembleur x64 `Vec4fAdd` et les deux extraits de désassemblage commentés.
- La suite de tests automatisés et les scénarios de benchmark reproductibles.
- Le fichier de résultats bruts `Benchmark/Benchmark.csv`.
- Le commentaire et la session de profiling CPU.
- Le README décrivant la compilation, l'exécution des tests et benchmarks, les conventions de l'API, les fonctionnalités, les limites et les sources utilisées.
