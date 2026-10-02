# Analisi della run PathMNIST 20261002T104242464569Z

La run completa i seed 42, 47 e 72 e supera la baseline, ma l'accuracy media
scende rispetto alla run 083530. Il protocollo è più completo: fonti recuperate
online e nove training di ablation, con configurazione finale comune ai seed.

## Verifiche effettuate

Ricalcolate da array storici accuracy, macro-F1, balanced accuracy, soglie,
coverage, accuracy selettiva, coda e AUROC per severità: tutti coincidono.
Il controllo nativo di integrità non segnala problemi nei tre seed, inclusi
i checkpoint referenziati dalle ablation. Come nella run precedente, l'audit
ricorsivo trova riferimenti al YAML originario del seed 42 nel frozen input
dei seed 47/72: sono ambiguità di provenienza, non hash errati dei file locali.
Gli esiti grezzi e supplementari sono in `audit.json` e `supplemental.json`.

Le metriche delle ablation sono quelle dei report; sono stati verificati i
checksum dei relativi checkpoint, non ripetute nove inferenze. Non è stato
eseguito un replay del training né riscritta la run.

## Ricerca online e configurazione

`research_online=true`, `require_research=true`, cache `literature_cache_online_01`.
Il primo seed recupera sei fonti `arxiv_api` (mode online); i successivi
riutilizzano quelle fonti (mode cache). Lo Scout produce sette idee per seed.
La provenienza online è dimostrata dagli artefatti, ma non prova il supporto
scientifico di ogni extrapolazione del reasoning.

`ablation_suite=true`, `search_per_seed=false`. La ricerca seleziona sul seed
42 una ResNet18 larga 48, dropout 0,15, lr 0,0003, weight decay 0,0001,
label smoothing 0,1, class weighting, AdamW/cosine, 15 epoche e batch 128.
La rappresentazione finale è standardizzazione più hflip e rotate180.
Configurazione e rappresentazione sono verificate uguali fra seed.

Rispetto alla precedente run cambiano insieme learning rate (0,001→0,0003),
decay (0,001→0,0001), dropout (0,1→0,15) e augmentation (brightness rimosso).
Non si può attribuire il peggioramento alla ricerca online: sono cambiate
più scelte e non è stato eseguito un confronto che isoli la letteratura.

## Classificazione

| Seed | Accuracy | Macro-F1 | Baseline accuracy | Delta baseline |
|---|---:|---:|---:|---:|
| 42 | 88,2730% | 0,866695 | 82,1309% | +6,1421 pp |
| 47 | 90,5571% | 0,880190 | 84,0251% | +6,5320 pp |
| 72 | 88,3426% | 0,856021 | 82,8134% | +5,5292 pp |
| Media | 89,0576% | 0,867635 | 82,9898% | +6,0678 pp |

Accuracy media precedente 92,6370%: differenza -3,5794 pp. Macro-F1 precedente
0,904868: differenza -0,037233. Deviazione standard popolazione accuracy
1,0607 pp, contro 0,5910 pp della run precedente. Tre seed non consentono
conclusioni forti sulla variabilità generale.

Validation finale 99,4602 / 99,5102 / 99,6401%; gap test rispettivamente
11,19 / 8,95 / 11,30 pp. La validation del trial selezionato è 99,4002%,
quasi identica a quella della precedente run, ma non predice il miglioramento
sul test. Stroma resta debole: recall 59,62 / 54,63 / 50,12%. Smooth muscle
ha precision 55,36 / 64,50 / 57,18%; adipose ha recall 71,45% sul seed 42.

## Ablation della rappresentazione

| Rappresentazione | Accuracy 42 | Accuracy 47 | Accuracy 72 | Media | Macro-F1 media |
|---|---:|---:|---:|---:|---:|
| Unit scaling, nessuna aug. | 88,4819% | 87,7994% | 87,2145% | 87,8319% | 0,845226 |
| Standardize, nessuna aug. | 88,7465% | 87,5348% | 87,9526% | 88,0780% | 0,848358 |
| Standardize + hflip | 89,0808% | 87,4095% | 89,4847% | 88,6583% | 0,864977 |
| Selezionata: standardize + hflip + rotate180 | 88,2730% | 90,5571% | 88,3426% | 89,0576% | 0,867635 |

In media la standardizzazione aggiunge 0,2460 pp rispetto a unit scaling;
hflip aggiunge 0,5804 pp rispetto a standardize-only; la rappresentazione
selezionata aggiunge 0,3993 pp rispetto a hflip-only. Il vantaggio non è uniforme:
hflip-only supera la selezionata sui seed 42 (+0,8078 pp) e 72 (+1,1421 pp);
la selezionata prevale nettamente sul seed 47 (+3,1476 pp).

Questi risultati non significano che il selettore abbia visto e ignorato tali
accuracy di test: la ricerca usa validation e le ablation sono misure successive.
Non usare il test per sostituire a posteriori il vincitore. Inoltre augmentation
materializzate cambiano gli esempi processati per epoca: gli effetti comprendono
anche il maggiore costo di training, non sono un confronto a compute identico.

## Astensione e OOD

| Seed | Coverage test | Accuracy accettati | Rinvii | AUROC OOD medio |
|---|---:|---:|---:|---:|
| 42 | 60,7939% | 95,4639% | 2.815 | 0,922349 |
| 47 | 65,9889% | 95,6311% | 2.442 | 0,776469 |
| 72 | 62,1309% | 95,7857% | 2.719 | 0,989675 |

La coverage e l'accuracy selettiva peggiorano rispetto alla precedente run.
La soglia resta calibrata all'80% sulla validation; non garantisce quell'obiettivo
sul test. I metodi selezionati sono max-softmax sui seed 42/47 ed entropia su 72.

Tutti i seed hanno `ood_pass=true`, ma il criterio richiede solo AUROC ≥0,5
a ogni severità dello score selezionato. Sul seed 47 a sigma 0,4 l'AUROC
max-softmax è 0,558075, vicino al caso; l'entropia alternativa è 0,492486.
Alla soglia pulita il 64,44% dei casi corrotti fortemente è autoaccettato.
Le quote corrispondenti sui seed 42 e 72 sono 0%; sul seed 47 al rumore medio
è autoaccettato il 4,16%. Non è quindi risolta la stabilità OOD fra seed.
Il rumore dipende dal seed e rimane un proxy di corruzione, non OOD clinico.

## Accettazione e prossimi passi

Ora passano web retrieval, ablation e ripetizioni congelate. Tutti i seed sono
tecnicamente completi ma `acceptance_ready=false`, `trl7_evidence_complete=false`.
Restano pending fault injection sul codice esatto, replay completo e sign-off
indipendente; falliscono le mitigazioni approvate dei rilievi. La run non
dimostra ancora il contributo isolato del reasoning rispetto a una ricerca
convenzionale a stesso spazio e budget.

Priorità: mantenere immutate entrambe le run; confrontare le loro configurazioni
con protocollo validation-only; analizzare perché la nuova scelta mantiene
validation elevata ma perde generalizzazione; ripetere l'analisi OOD con
corruzioni comuni ai seed e criterio esplicito di score non idoneo. Le ablation
forniscono ora evidenza utile, ma nessuna combinazione è uniformemente vincente.
