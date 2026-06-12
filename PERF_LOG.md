# Journal des Performances et Suivi du Code : Micro-Beamforming

* **Date de référence :** 21 mai 2026
* **Configuration de test :** Serveur SRV07
* **Données de test :** 1 Frame, 960 Éléments, 30 SAPs au total

## 1. Phase d'Exploration : Journal des Performances Initiales

Ce premier tableau retrace les premiers essais du stage. Les temps ont été mesurés en lançant le script une seule fois, sans exécution à vide au démarrage (pas de warm-up). Ce tableau prend donc en compte le temps que met Python à charger les bibliothèques et à compiler le code la toute première fois.

| Version du Code | Environnement | Technologie / Bibliothèque | Temps de Calcul | Speedup (vs Baseline Python) | Speedup Global (vs MATLAB) | Réduction du Temps (vs Baseline) | Analyse technique et modifications apportées | Étape suivante du stage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`beamform_sorbet_micro.m`** | MATLAB | Code original de référence | **28.87 s** | *N/A* | *Référence absolue* | *N/A* | Code de base fourni par le laboratoire pour vérifier que les futures versions Python donnent les mêmes résultats. | Point de départ de l'étude. |
| **`beamform_sorbet_micro.py`**<br>*(Baseline)* | Python (CPU) | [`scipy.interpolate.interp1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html) | **19.45 s** | **1.00x** *(Réf)* | **1.48x** | **0.00 %** | Traduction ligne par ligne du code MATLAB en Python. Les résultats sont validés par un calcul de corrélation ($0.999995$). | Lancement d'un premier profilage avec cProfile pour voir quelles fonctions prennent du temps. |
| **Profilage de la Baseline** | Analyse CPU | [`cProfile`](https://docs.python.org/3/library/profile.html) + [`matplotlib`](https://matplotlib.org/) | *N/A (Analyse)* | *N/A* | *N/A* | *N/A* | L'analyse montre que la fonction d'interpolation de SciPy (`__init__`, `__call__`) prend plus de 7 secondes à elle seule à cause de la création répétée d'objets. | Remplacement de l'interpolation SciPy par la fonction `numpy.interp` pour simplifier. |
| **Optimisation 1** | Python (CPU) | Remplacement par [`numpy.interp`](https://numpy.org/doc/stable/reference/generated/numpy.interp.html) | **15.90 s** | **1.22x** | **1.82x** | **18.25 %** | Le calcul direct avec NumPy évite la création d'objets lourds. Le temps descend à 15.90 s. | Lancement d'un nouveau profilage pour voir ce qui bloque encore. |
| **Profilage de l'Optimisation 1** | Analyse CPU | Profilage incrémental (`cProfile` + CSV) | *N/A (Analyse)* | *N/A* | *N/A* | *N/A* | SciPy a disparu des fonctions coûteuses. Le problème vient maintenant de la boucle `for` principale en Python qui se répète 30 720 fois et prend 12.67 s. | Essai de parallélisation de la boucle sur plusieurs cœurs CPU avec des Threads. |
| **Optimisation 2** | Python (CPU Multi-threads) | [`concurrent.futures.ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html) | **18.92 s** | **0.86x** *(Régression)* | **1.53x** | **-34.46 %** | Le temps augmente car les threads passent leur temps à s'attendre à cause du verrou interne de Python (le GIL). Cette limite classique de Python pour le calcul intensif est expliquée dans l'atelier du [Centre de Calcul du CNES](https://sourcesup.renater.fr/wiki/atelieromp/_media/20221201_atelier_omp_optimisation_code_python.pdf). | Le multi-threading classique ne fonctionne pas ici. Essai de la compilation JIT avec Numba qui permet de contourner ce verrou. |
| **Profilage du Multi-threading** | Analyse CPU Parallèle | `cProfile` dédié Threads + CSV | *N/A (Analyse)* | *N/A* | *N/A* | *N/A* | Les fonctions de blocage des threads (`wait` et `lock.acquire`) cumulent plus de 7 secondes d'attente inutile pour le processeur. | Confirmation du problème du GIL. Passage au décorateur `@njit(parallel=True)`. |
| **Optimisation 3** | Python (CPU JIT Parallèle) | [`numba` (`@njit(parallel=True)`)](https://numba.readthedocs.io/en/stable/user/parallel.html) | **30.94 s** | **0.63x** *(Régression)* | **0.93x** | **-59.07 %** | Le temps augmente fortement sur ce run unique. Cela s'explique par le fait que Numba doit compiler le code lors du tout premier lancement, ce qui fausse la mesure. | Utilisation des outils de diagnostic de Numba pour comprendre si la parallélisation a fonctionné. |
| **Profilage & Diagnostic Numba** | Analyse CPU | [`parallel_diagnostics(level=4)`](https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics) | *N/A (Analyse)* | *N/A* | *N/A* | *N/A* | Le rapport indique que la fusion automatique des boucles a échoué (`fusion failed`). Créer trop de petits tableaux à chaque itération sature la mémoire RAM. | Les limites du processeur étant atteintes sur cette boucle, l'étude s'oriente vers le calcul sur **GPU**. |
| **Étude de l'état de l'art** | Analyse Théorique | Outils de calcul GPU (Présentations CNRS) | *N/A (Analyse)* | *N/A* | *N/A* | *N/A* | Recherche des outils pour utiliser le GPU en Python à partir du [Webinaire de la Gray Scott School (CNRS)](https://www.youtube.com/watch?v=4RsXXTCHzLo). Étude de JAX, cuPyNumeric et CuPy. | Choix de la bibliothèque **CuPy** car ses fonctions s'écrivent presque de la même façon que NumPy. |
| **Optimisation 5** | Python (GPU Massif) | [`cupy` (Précision `float32`)](https://cupy.dev/) | **3.74 s** | **5.20x** | **7.72x** | **80.77 %** | Passage des calculs sur la carte graphique and conversion des données en précision simple (`float32`), ce qui accélère fortement le traitement. | Le calcul sur GPU fonctionne et les résultats sont justes. Profilage des fonctions GPU pour optimiser. |
| **Étape de Profilage** | Analyse CPU / GPU | `cProfile` + Graphique des coûts (PNG) | *Analyse* | *N/A* | *N/A* | *N/A* | Le chronomètre montre que l'envoi des données vers le GPU (`cupy.asarray`) prend le plus de temps (2.159 s), alors que le calcul de l'interpolation est très rapide (0.408 s). | Essai d'utilisation des flux asynchrones (Streams CUDA) pour envoyer et calculer les données en même temps. |
| **Optimisation Finalisée** | Python (GPU Asynchrone) | [`cupy.cuda.Stream`](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html) | **2.60 s** | **7.48x** | **11.10x** | **90.99 %** | Utilisation des Streams pour commencer à calculer un signal pendant que le signal suivant est en train d'être transféré sur la carte. | Nettoyage des fonctions à l'intérieur de la boucle pour éviter de recréer des tableaux inutilement. |
| **Optimisation VRAM Fine** | Python (GPU Asynchrone v2) | [`cp.ascontiguousarray`](https://docs.cupy.dev/en/stable/reference/generated/cupy.ascontiguousarray.html) + Nettoyage Boucle | **2.42 s** | **8.08x** | **11.93x** | **91.62 %** | Sortie de la fonction `arange` en dehors de la boucle pour qu'elle ne soit créée qu'une seule fois. Rangement des données côte à côte en mémoire pour accélérer les accès. | Fin de la partie recherche sur une seule carte graphique. |
| **Évaluation Multi-GPU (Piste 2 - Essai A)** | Windows (Local) | `conda install` (Outil [`cuPyNumeric`](https://github.com/nv-legate/cupynumeric)) | **N/A (Bloqué)** | *N/A* | *N/A* | *N/A* | Impossible d'installer `cuPyNumeric` sur Windows car cet outil nécessite obligatoirement un système d'exploitation Linux natif. | Tentative d'installation dans un système Linux virtuel (WSL2) installé sur l'ordinateur. |
| **Évaluation Multi-GPU (Piste 2 - Essai B)** | Linux (WSL2 Virtualisé) | [`cuPyNumeric`](https://github.com/nv-legate/cupynumeric) sur Miniforge Linux | **ABANDON (Incompatible)** | *N/A* | *N/A* | *N/A* | L'outil a provoqué des plantages mémoires dans l'environnement virtuel (`CUDA_ERROR_OUT_OF_MEMORY`). De plus, il demande Python 3.10+ alors que Vermon utilise Python 3.9 pour rester compatible avec MATLAB. | Piste abandonnée car l'outil est fait pour des réseaux de plusieurs ordinateurs (Clusters), ce qui dépasse le besoin du stage. |
| **Évaluation Multi-GPU (Piste 3)** | Windows (Local) | Bibliothèque [`JAX`](https://jax.readthedocs.io/) | **ÉCHEC (Syntaxe)** | *N/A* | *N/A* | *N/A* | Erreur au lancement (`TypeError: object does not support item assignment`) car JAX interdit de modifier directement les cases d'un tableau créé. | Piste écartée car l'utilisation de JAX aurait demandé de réécrire complètement toute la logique de l'algorithme. |
| **Perspectives d'Avenir** | Python / C++ (GPU Avancé) | Kernels CUDA personnalisés ([`CuPy RawKernel`](https://docs.cupy.dev/en/stable/user_guide/kernel.html) ou [`Numba CUDA`](https://numba.readthedocs.io/en/stable/cuda/index.html)) | *À déterminer* | *N/A* | *N/A* | *N/A* | Piste pour la suite : regrouper le calcul des indices, les décalages physiques et l'interpolation dans une seule fonction écrite directement en langage CUDA pour éviter les allers-retours mémoire. | Idée d'ouverture pour la fin du rapport si le temps le permet. |

---

## 2. Validation des Méthodes de Mesure Temporelle

Pour assurer la validité des métriques, une étude comparative des outils de chronométrage intégrés à Python a été menée afin de sélectionner l'indicateur le plus stable face aux variations de l'ordinateur d'exploitation.

| Outil de mesure | Bibliothèque / Documentation | Validé (Oui/Non) | Raison du choix |
| :--- | :--- | :--- | :--- |
| **`time.time()`** | [`time` (officiel)](https://docs.python.org/3/library/time.html#time.time) | **OUI (testé)** | Donne l'heure de l'ordinateur. Cette méthode peut manquer de précision si l'horloge système se synchronise ou s'ajuste pendant que le code tourne. |
| **`timeit.timeit()`** | [`timeit` (officiel)](https://docs.python.org/3/library/timeit.html) | **NON (écarté)** | Fait pour mesurer des toutes petites lignes de code isolées, pas pour chronométrer un algorithme complet de reconstruction d'image. |
| **`time.process_time()`** | [`time` (officiel)](https://docs.python.org/3/library/time.html#time.process_time) | **NON (écarté)** | Ne mesure que le temps passé par le processeur principal (CPU). Il ne prend pas en compte le temps passé par la carte graphique (GPU) à faire les calculs. |
| **`time.perf_counter()`** | [`time` (officiel)](https://docs.python.org/3/library/time.html#time.perf_counter) | **OUI (retenu - choix final)** | Utilise un compteur interne très précis lié au processeur qui avance de façon régulière. C'est l'outil standard pour mesurer une durée en Python. |

### Calculs utilisés pour les indicateurs de performance

* **Speedup Global (par rapport à MATLAB) :** Calcule combien de fois la version Python est plus rapide que le code de base du laboratoire.
  $$\text{Speedup}_{\text{global}} = \frac{T_{\text{MATLAB}}}{T_{\text{Version}}}$$

* **Gain vs Baseline Python :** Permet de voir le gain apporté par une modification précise par rapport au tout premier code Python écrit.
  $$\text{Gain}_{\text{vs\_baseline}} = \frac{T_{\text{Baseline Python}}}{T_{\text{Version Optimisée}}}$$

* **Taux de réduction du temps de calcul :** Donne le pourcentage de temps économisé grâce à l'optimisation.
  $$\text{Réduction} = \frac{T_{\text{Initial}} - T_{\text{Optimisé}}}{T_{\text{Initial}}} \times 100$$

---

## 3. Ressources Documentaires et Veille Technique

Cette section rassemble les documents, liens officiels et présentations d'ingénieurs utilisés pendant le stage pour comprendre les blocages techniques rencontrés.

### Fonctionnement des threads en Python
* **Le mécanisme du GIL (Global Interpreter Lock) :** C'est une règle interne de Python qui interdit à plusieurs threads d'exécuter du code en même temps. Cela explique pourquoi la version multi-threads a été plus lente que la version de base pour notre calcul numérique intensif.  
  *Référence :* [Documentation officielle du GIL Python](https://docs.python.org/3/glossary.html#term-GIL)

### Méthodes pour mesurer et optimiser le code
* **Exemple de boucle de benchmark du CNES :** Le protocole mis en place pour nettoyer les mesures s'inspire du code d'exemple proposé par leur atelier de calcul haute performance.  
  *Référence :* [Support de cours du Centre de Calcul du CNES (Atelier Numérique OMP)](https://sourcesup.renater.fr/wiki/atelieromp/_media/20221201_atelier_omp_optimisation_code_python.pdf)
 * **Documentation officielle du module de mesure Python (`timeit`) :** Guide officiel expliquant pourquoi la moyenne brute et l'écart-type sont des indicateurs peu utiles en informatique de performance, et pourquoi la valeur minimale (`min()`) reste la seule mesure scientifique fiable pour évaluer un algorithme face aux perturbations du système d'exploitation.  
  *Référence :* [Documentation officielle Python - Module timeit (Note sur le Minimum)](https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat)
* **Guide de propreté du code :** Conseils suivis pour organiser le script, enlever les morceaux de code inutiles et rendre l'ensemble plus lisible.  
  *Référence :* [Analyse et Optimisation de code Python - S. Robert](https://blog.stephane-robert.info/docs/developper/programmation/python/linting/)

### Documentations des outils testés pour le Multi-GPU
* **Bibliothèque cuPyNumeric de NVIDIA :** Analyse des fichiers d'installation de cet outil conçu pour distribuer les calculs NumPy sur plusieurs cartes graphiques en réseau. Testé sous Windows et abandonné suite aux incompatibilités avec l'environnement de l'entreprise.  
  *Référence :* [Dépôt officiel cuPyNumeric sur GitHub](https://github.com/nv-legate/cupynumeric)

---

## 4. Réorganisation du Code et Bilan Final des Performances

Après avoir testé les différentes technologies, les différents scripts de recherche ont été regroupés dans une structure de fichiers plus propre et plus simple à utiliser pour l'équipe.

Le code a été séparé en deux fichiers :
1. **`src/beamformer_engines.py` :** Contient uniquement les fonctions de calcul pur (les blocs Numba `@njit` et les fonctions d'interpolation de CuPy).
2. **`src/beamformer.py` :** Contient une classe unique appelée `Beamformer`. L'utilisateur a juste besoin de créer cet objet avec les paramètres de sa sonde, puis de lancer la méthode `.das(rf, method='...')` en choisissant le moteur de calcul qu'il veut tester (CPU, Numba ou GPU).

### Protocole de mesure retenu (Inspiré du CNES)

Pour éviter que les tâches de fond de l'ordinateur ne faussent les résultats (comme lors du premier tableau), le script final utilise un protocole inspiré de la [page 38 du document d'optimisation du CNES](https://sourcesup.renater.fr/wiki/atelieromp/_media/20221201_atelier_omp_optimisation_code_python.pdf#page=38) :
* **Un run de préchauffage (Warm-up) :** Le code est lancé une première fois à vide sans être chronométré. Cela permet à Numba de faire sa compilation et à CuPy d'initialiser la carte graphique.
* **Une boucle de 5 répétitions :** Le calcul est répété 5 fois de suite.
* **Sélection du temps minimum :** Le script garde uniquement la valeur la plus basse. Contrairement à la moyenne qui intègre les ralentissements accidentels du système d'exploitation, le minimum permet de mesurer la vitesse réelle de l'algorithme lorsque la machine est totalement disponible.
### Tableau Comparatif après Réorganisation du Code

Grâce à la centralisation de l'architecture, à la suppression des allocations de tableaux répétitives dans les boucles (comme le déplacement du `arange`) et au nettoyage des variables intermédiaires, les temps de traitement ont été réduits sur tous les moteurs de calcul.

Voici les temps mesurés sur le **Serveur SRV07 (1 Frame, 960 Éléments, 30 SAPs au total)** :

| Rang | Moteur de Calcul (Option `method`) | Environnement Matériel | Temps d'Exécution (s) | Speedup vs MATLAB (Référence) | Gain vs Baseline Python (SciPy) | Réduction du Temps / MATLAB (%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| *Réf* | **MATLAB (Code original)** | CPU Standard | **28.87 s** | *1.00x (Référence)* | *N/A* | *Référence* |
| **#1** | **`gpu_block`** | GPU (NVIDIA CUDA Standard) | **1.67 s** | **17.31x** | **6.33x** | **-94.22 %** |
| **#2** | **`gpu_streams`** | GPU (NVIDIA CUDA + Flux Asynchrones) | **1.71 s** | **16.90x** | **6.18x** | **-94.08 %** |
| **#3** | **`numba`** | CPU (Compilation JIT Parallèle) | **8.15 s** | **3.54x** | **1.29x** | **-71.75 %** |
| **#4** | **`numpy`** | CPU (Vectorisation native) | **8.70 s** | **3.32x** | **1.21x** | **-69.85 %** |
| **#5** | **`threads`** | CPU (Multi-threading 8 workers) | **9.25 s** | **3.12x** | **1.14x** | **-67.95 %** |
| **#6** | **`scipy`** | CPU (Mono-thread après nettoyage de la structure) | **10.55 s** | **2.74x** | **1.00x (Ref)** | **-63.45 %** |

### Analyse et Interprétation des Résultats

1. **Impact du nettoyage de la structure :** La version de base `scipy` passe de 19.45 s dans le premier tableau à 10.55 s ici. Le fait d'avoir réorganisé le code sous forme de classe évite à Python de recréer et de recharger les dictionnaires de paramètres à chaque étape du calcul.
2. **Correction du temps Numba :** La phase de warm-up ayant retiré le temps de première compilation, Numba affiche désormais sa vitesse réelle sur le processeur (**8.15 s**), passant comme prévu devant la version NumPy standard (8.70 s).
3. **Comparaison des moteurs GPU :** Le mode bloc (1.67 s) reste très proche du mode streams (1.71 s). Sur le volume de test d'une seule image (1 Frame), le coût fixe de gestion des flux asynchrones annule l'avantage du transfert en tâche de fond. Envoyer les données d'un seul bloc est ici la solution la plus directe.
4. **Validation des données :** Toutes les méthodes intégrées à la classe ont été contrôlées. Les matrices de sortie sont parfaitement conformes aux images générées par MATLAB, avec un coefficient de corrélation finale mesuré à **0.99995**.

L'implémentation finale sur l'architecture GPU permet de répondre aux objectifs fixés par Vermon dans le cadre de ce stage, en associant une structure de code modulaire à un gain de performance global de **17.31x** par rapport à la version d'origine.
