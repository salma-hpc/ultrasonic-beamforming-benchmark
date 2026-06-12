# Journal des Performances et Suivi du Code : Micro-Beamforming

- **Date de référence :** 21 mai 2026
- **Configuration de test :** Serveur SRV07
- **Données de test :** 1 Frame, 960 Éléments, 30 SAPs au total

## 1. Phase d’Exploration : Journal des Performances Initiales

Ce premier tableau retrace les premiers essais du stage. Les temps ont été mesurés en lançant le script une seule fois, sans exécution à vide au démarrage (pas de warm-up). Ce tableau prend donc en compte le temps que met Python à charger les bibliothèques et à compiler le code la toute première fois.

| Version | Env. | Techno | Temps | vs Base | vs MATLAB | Gain % |
|---|---|---|---:|---:|---:|---:|
| `beamform_sorbet_micro.m` | MATLAB | Réf. | 28.87 s | N/A | Réf. | N/A |
| `beamform_sorbet_micro.py`<br>*(Base)* | Py CPU | `scipy.interp1d` | 19.45 s | 1.00x | 1.48x | 0.00 % |
| Profilage base | Ana. CPU | `cProfile` + `matplotlib` | N/A | N/A | N/A | N/A |
| Opt. 1 | Py CPU | `numpy.interp` | 15.90 s | 1.22x | 1.82x | 18.25 % |
| Profilage 1 | Ana. CPU | `cProfile` + CSV | N/A | N/A | N/A | N/A |
| Opt. 2 | Py CPU MT | `ThreadPoolExecutor` | 18.92 s | 0.86x | 1.53x | -34.46 % |
| Profilage MT | Ana. CPU | `cProfile` + CSV | N/A | N/A | N/A | N/A |
| Opt. 3 | Py JIT | `numba` | 30.94 s | 0.63x | 0.93x | -59.07 % |
| Diag. Numba | Ana. CPU | `parallel_diagnostics` | N/A | N/A | N/A | N/A |
| Opt. GPU | Py GPU | `cupy` float32 | 3.74 s | 5.20x | 7.72x | 80.77 % |
| Profilage GPU | Ana. CPU/GPU | `cProfile` + PNG | N/A | N/A | N/A | N/A |
| Final | Py GPU async | `cupy.cuda.Stream` | 2.60 s | 7.48x | 11.10x | 90.99 % |
| VRAM | Py GPU async v2 | `cp.ascontiguousarray` | 2.42 s | 8.08x | 11.93x | 91.62 % |
| Multi-GPU A | Windows | `cuPyNumeric` | Bloqué | N/A | N/A | N/A |
| Multi-GPU B | Linux WSL2 | `cuPyNumeric` | Abandon | N/A | N/A | N/A |
| Multi-GPU C | Windows | `JAX` | Échec | N/A | N/A | N/A |
| Perspectives | Python / C++ | Kernels CUDA custom | À déterminer | N/A | N/A | N/A |

### Détails techniques

| Version | Analyse | Suite |
|---|---|---|
| `beamform_sorbet_micro.m` | Code original de référence fourni par le laboratoire. | Point de départ de l’étude. |
| `beamform_sorbet_micro.py`<br>*(Base)* | Traduction ligne par ligne du code MATLAB en Python. Corrélation : 0.999995. | Premier profilage avec `cProfile`. |
| Profilage base | La fonction d’interpolation SciPy prend plus de 7 s à cause de la création répétée d’objets. | Remplacement par `numpy.interp`. |
| Opt. 1 | Le calcul direct avec NumPy évite les objets lourds. | Nouveau profilage. |
| Profilage 1 | La boucle `for` principale en Python se répète 30 720 fois et prend 12.67 s. | Essai de parallélisation avec Threads. |
| Opt. 2 | Les threads se bloquent à cause du GIL. | Essai de Numba. |
| Profilage MT | Les fonctions de blocage des threads cumulent plus de 7 s d’attente. | Passage à `@njit(parallel=True)`. |
| Opt. 3 | Le premier lancement inclut le temps de compilation Numba. | Utilisation de `parallel_diagnostics`. |
| Diag. Numba | La fusion automatique des boucles a échoué et la mémoire RAM est saturée par trop de petits tableaux. | Passage au GPU. |
| Étude de l’état de l’art | Étude de JAX, cuPyNumeric et CuPy pour le calcul GPU. | Choix de CuPy. |
| Opt. GPU | Passage des calculs sur la carte graphique avec précision simple (`float32`). | Profilage GPU. |
| Profilage GPU | L’envoi des données vers le GPU (`cupy.asarray`) prend le plus de temps. | Essai des streams CUDA. |
| Final | Utilisation des Streams pour faire calcul et transfert en parallèle. | Nettoyage des boucles. |
| VRAM | `arange` sorti de la boucle et données contiguës en mémoire. | Fin de l’étude mono-GPU. |
| Multi-GPU A | `cuPyNumeric` impossible à installer sur Windows natif. | Test sous WSL2. |
| Multi-GPU B | `cuPyNumeric` provoque des erreurs mémoire sous WSL2 et demande Python 3.10+. | Piste abandonnée. |
| Multi-GPU C | `JAX` interdit la modification directe des tableaux. | Piste écartée. |
| Perspectives | Idée de regrouper indices, décalages et interpolation dans un kernel CUDA unique. | Ouverture pour la suite. |

## 2. Validation des méthodes de mesure temporelle

Pour assurer la validité des métriques, une étude comparative des outils de chronométrage intégrés à Python a été menée afin de sélectionner l’indicateur le plus stable face aux variations de l’ordinateur d’exploitation.

| Outil | Doc. | Validé | Raison |
|---|---|---|---|
| `time.time()` | [`time`](https://docs.python.org/3/library/time.html#time.time) | Oui | Donne l’heure système, mais peut être affecté par une resynchronisation de l’horloge. |
| `timeit.timeit()` | [`timeit`](https://docs.python.org/3/library/timeit.html) | Non | Adapté aux petites portions de code, pas à un algorithme complet. |
| `time.process_time()` | [`time`](https://docs.python.org/3/library/time.html#time.process_time) | Non | Ne mesure pas le temps GPU. |
| `time.perf_counter()` | [`time`](https://docs.python.org/3/library/time.html#time.perf_counter) | Oui | Compteur précis et monotone, retenu comme choix final. |

### Calculs utilisés pour les indicateurs de performance

- **Speedup global (vs MATLAB) :**
\[
\text{Speedup}_{\text{global}} = \frac{T_{\text{MATLAB}}}{T_{\text{Version}}}
\]

- **Gain vs Baseline Python :**
\[
\text{Gain}_{\text{vs\_baseline}} = \frac{T_{\text{Baseline Python}}}{T_{\text{Version Optimisée}}}
\]

- **Réduction du temps de calcul :**
\[
\text{Réduction} = \frac{T_{\text{Initial}} - T_{\text{Optimisé}}}{T_{\text{Initial}}} \times 100
\]

## 3. Ressources documentaires et veille technique

Cette section rassemble les documents, liens officiels et présentations d’ingénieurs utilisés pendant le stage pour comprendre les blocages techniques rencontrés.

### Fonctionnement des threads en Python
- **GIL (Global Interpreter Lock)** : mécanisme interne de Python qui empêche plusieurs threads d’exécuter du bytecode Python en parallèle.  
  Référence : [Documentation officielle du GIL Python](https://docs.python.org/3/glossary.html#term-GIL)

### Mesure et optimisation
- **Boucle de benchmark du CNES** : le protocole utilisé s’inspire de l’atelier HPC du CNES.  
  Référence : [Support CNES](https://sourcesup.renater.fr/wiki/atelieromp/_media/20221201_atelier_omp_optimisation_code_python.pdf)
- **Documentation `timeit`** : explique pourquoi le minimum est souvent la métrique la plus fiable pour l’évaluation de performance.  
  Référence : [Documentation Python `timeit`](https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat)
- **Guide de propreté du code** : utilisé pour le nettoyage et la réorganisation du script.  
  Référence : [Guide S. Robert](https://blog.stephane-robert.info/docs/developper/programmation/python/linting/)

### Outils testés pour le Multi-GPU
- **cuPyNumeric** : outil étudié pour la distribution de calculs sur plusieurs GPU.  
  Référence : [Dépôt officiel cuPyNumeric](https://github.com/nv-legate/cupynumeric)

## 4. Réorganisation du code et bilan final des performances

Après les essais exploratoires, les scripts ont été regroupés dans une structure plus simple et plus propre pour l’équipe.

Le code a été séparé en deux fichiers :
1. `src/beamformer_engines.py` : fonctions de calcul pur.
2. `src/beamformer.py` : classe `Beamformer` avec appel `.das(rf, method='...')`.

### Protocole de mesure retenu
Le script final utilise un protocole inspiré du CNES :
- Un warm-up non chronométré.
- Une boucle de 5 répétitions.
- Conservation du temps minimum.

### Tableau comparatif après réorganisation du code

Grâce à la centralisation de l’architecture, à la suppression des allocations répétitives dans les boucles et au nettoyage des variables intermédiaires, les temps de traitement ont été réduits sur tous les moteurs de calcul.

| Rang | Moteur | Matériel | Temps | Speedup MATLAB | Gain vs Base | Réduction vs MATLAB |
|---|---|---|---:|---:|---:|---:|
| Réf | MATLAB | CPU standard | 28.87 s | 1.00x | N/A | Réf. |
| #1 | `gpu_block` | GPU CUDA standard | 1.67 s | 17.31x | 6.33x | -94.22 % |
| #2 | `gpu_streams` | GPU CUDA + streams | 1.71 s | 16.90x | 6.18x | -94.08 % |
| #3 | `numba` | CPU JIT parallèle | 8.15 s | 3.54x | 1.29x | -71.75 % |
| #4 | `numpy` | CPU vectorisé | 8.70 s | 3.32x | 1.21x | -69.85 % |
| #5 | `threads` | CPU multi-threading | 9.25 s | 3.12x | 1.14x | -67.95 % |
| #6 | `scipy` | CPU mono-thread | 10.55 s | 2.74x | 1.00x | -63.45 % |

### Analyse et interprétation des résultats

1. **Impact du nettoyage de la structure** : la version `scipy` passe de 19.45 s à 10.55 s grâce à la réorganisation du code.
2. **Correction du temps Numba** : le warm-up retire le coût de la compilation initiale.
3. **Comparaison GPU** : `gpu_block` reste légèrement devant `gpu_streams` sur une seule frame.
4. **Validation des données** : toutes les méthodes produisent des sorties conformes à MATLAB avec une corrélation finale de 0.99995.

L’implémentation finale GPU répond aux objectifs du stage avec un gain global de 17.31x par rapport à la version d’origine.
