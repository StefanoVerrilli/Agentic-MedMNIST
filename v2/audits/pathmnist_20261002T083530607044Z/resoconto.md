# Resoconto completo — PathMNIST 20261002T083530607044Z

Analisi del 2 ottobre 2026. Evidenze originali preservate; gli output dell'audit
sono in questa directory. Confronto: run `20261001T175019922973Z`.

## Giudizio

La run migliora classificazione e astensione e risolve il problema metodologico
delle configurazioni diverse fra seed. Il detector OOD rimane instabile anche
a configurazione di training congelata. Il contributo dell'orchestrazione non
è ancora isolato e l'accettazione finale resta aperta.

## Protocollo effettivamente eseguito

- Dataset completo: 89.996 train, 10.004 validation, 7.180 test; split ufficiali.
- Seed 42, 47, 72; ricerca solo sul primo seed (`search_per_seed=false`).
- Otto trial completati, zero fallimenti, cinque famiglie coperte.
- I seed 47 e 72 riprendono il vincitore senza adattarlo al test.
- Budget finale 15 epoche, batch 128, CUDA, determinismo Torch dichiarato.
- Ollama `qwen3.8`, richiesto; 20 decisioni sul primo seed, 17 per seed successivo.
- `ablation_suite=false`, `research_online=false`, `validation_evidence=null`.
- Durata complessiva registrata circa 64 minuti e 43 secondi; somma dei tempi
  dei trial 854,45 secondi, circa 14 minuti e 14 secondi. Sono tempi registrati,
  non misure strumentate di GPU-secondi; il restante tempo non è attribuito
  automaticamente al solo LLM.

Modello congelato: SmallResNet18, larghezza iniziale 48, dropout 0,1, AdamW,
learning rate 0,001, weight decay 0,001, cosine scheduler, label smoothing 0,1,
class weighting train-only. Standardizzazione train-only; augmentation finali
`hflip`, `rotate180`, `brightness`. Il parametro generico `depth=3` è inattivo
per questa famiglia. La nuova rationale lo riconosce correttamente.

Il modello ha 6.288.057 parametri. La baseline ha 5.433 parametri, usa
standardizzazione senza augmentation e lo stesso budget finale di epoche.
La rappresentazione materializza quattro versioni per immagine train: stessa
quantità di epoche non implica lo stesso numero di esempi processati.
`brightness` è una trasformazione deterministica di +10%, non jitter casuale
né una validazione di robustezza a differenti protocolli di colorazione.

## Risultati di classificazione

| Seed | Accuracy | Macro-F1 | Balanced accuracy | Baseline accuracy | Delta baseline |
|---|---:|---:|---:|---:|---:|
| 42 | 92,2981% | 0,898925 | 0,901332 | 82,1309% | +10,1672 pp |
| 47 | 92,1448% | 0,898564 | 0,897181 | 84,0251% | +8,1197 pp |
| 72 | 93,4680% | 0,917116 | 0,915932 | 82,8134% | +10,6546 pp |
| Media | 92,6370% | 0,904868 | 0,904815 | 82,9898% | +9,6472 pp |

Deviazione standard accuracy: 0,5910 pp popolazione (summary), 0,7238 pp
campionaria (analisi supplementare). Tre seed sono un campione piccolo e i
risultati condividono lo stesso test; non sono tre coorti indipendenti.

La vecchia run aveva accuracy media 88,9368% e macro-F1 media 0,863785.
Il miglioramento è +3,7002 pp di accuracy e +0,041083 di macro-F1. Gli errori
totali sulle tre valutazioni passano da 2.384 a 1.586 (798 in meno, -33,47%).
Questa è una comparazione descrittiva sul benchmark già osservato, non la
dimostrazione causale dell'efficacia delle singole modifiche.

Validation finale migliore: 99,4702%, 99,6002%, 99,5602%. Il gap test è ancora
7,17 / 7,46 / 6,09 pp. Il valore 99,4002% ripetuto nel summary dei tre seed
è invece la validation del trial di selezione del seed 42, non la misura finale
di ogni training. Best epoch: 15 / 15 / 14; tutti completano 15 epoche.
Raggiungere l'ultima epoca non dimostra da solo che serva aumentare il budget.

La ricerca confronta candidati a budget diversi (3, 6, 9 epoche); la selezione
finale considera il budget massimo. Gli esiti non costituiscono un confronto
equo fra famiglie: i transformer non ricevono lo stesso budget finale ResNet.

## Risultati per classe

| Classe | Recall medio | F1 medio | F1 medio precedente |
|---|---:|---:|---:|
| Adipose | 93,40% | 0,9390 | 0,8756 |
| Background | 100,00% | 0,9792 | 0,9366 |
| Debris | 91,25% | 0,8627 | 0,8002 |
| Lymphocytes | 99,84% | 0,9769 | 0,9740 |
| Mucus | 92,50% | 0,9450 | 0,9422 |
| Smooth muscle | 83,56% | 0,7949 | 0,6945 |
| Normal colon mucosa | 96,31% | 0,9596 | 0,9503 |
| Cancer-associated stroma | 60,97% | 0,7328 | 0,6470 |
| Adenocarcinoma epithelium | 96,51% | 0,9538 | 0,9534 |

Il maggior recupero di recall riguarda adipose: 80,29% → 93,40%. Stroma migliora
da 52,34% a 60,97%, ma rimane il punto debole: recall individuale
61,9952% / 56,5321% / 64,3705%. Molti errori finiscono in debris e smooth muscle.
Smooth muscle migliora soprattutto in precision/F1, non in recall medio
(84,68% → 83,56%). Non affermare un miglioramento uniforme di tutte le metriche.

## Astensione sui dati puliti

| Seed | Score selezionato su validation | Coverage test | Accuracy accettati | Casi da revisionare |
|---|---|---:|---:|---:|
| 42 | Entropia predittiva | 67,3677% | 97,4158% | 2.343 |
| 47 | Max-softmax | 65,2925% | 97,5683% | 2.492 |
| 72 | Max-softmax | 68,7047% | 97,6688% | 2.247 |

Rispetto alla run precedente aumenta la coverage media (62,17% → 67,12%) e
l'accuracy media accettata (96,50% → 97,55%). L'80% è il target sulla validation,
non una garanzia sul test. Il rischio residuo dei casi accettati è 2,33–2,58%.
Le code sono persistite, ma non dimostrano che la revisione umana sia conclusa.
Score e soglia vengono ricalibrati per seed sulla validation; la configurazione
congelata riguarda il training, non un unico score o una soglia comune.

## OOD: il limite più serio

AUROC del metodo selezionato sulla validation:

| Seed | Sigma 0,15 | Sigma 0,25 | Sigma 0,4 | Media | Esito attuale |
|---|---:|---:|---:|---:|---|
| 42, entropia | 0,962310 | 0,970555 | 0,946572 | 0,959812 | Pass |
| 47, max-softmax | 0,878883 | 0,300678 | 0,001771 | 0,393777 | Fail |
| 72, max-softmax | 0,919512 | 0,902798 | 0,577791 | 0,800034 | Pass |

Nel seed 47 l'inversione è già presente sulla validation (AUROC aggregato
0,424979 e 0,000817 alla severità forte); il selettore sceglie semplicemente
il migliore fra due score entrambi insufficienti. Non è soltanto un problema
di trasferimento della calibration al test.

Alla severità 0,4 tutti i 7.180 campioni sono predetti background in ogni seed.
Per il seed 47 la confidenza max-softmax media è 98,38% e 7.179/7.180 casi
(99,986%) superano la soglia di accettazione pulita. Alla severità 0,25 vengono
accettati il 75,58% dei casi. Sul seed 72, pur con pass, al rumore forte è
accettato il 50,40% dei casi. Sul seed 42 la quota scende allo 0,52%.

Il pass richiede solamente che ogni AUROC dello score selezionato sia almeno
0,5: non misura un tasso operativo accettabile di rifiuto. AUROC e comportamento
alla soglia sono due verifiche distinte. Queste corruzioni sono un proxy, non
una dimostrazione di OOD clinico. Inoltre il rumore dipende dal seed: la
variabilità include pesi del modello e realizzazioni diverse della corruzione.
Per isolare la stabilità del detector serve un pannello di rumore comune.

## Integrità e riproducibilità

Le metriche degli array storici sono tutte ricalcolate e confermate. Il controllo
di integrità nativo del progetto non segnala problemi nei tre seed.
L'audit ricorsivo più ampio segnala due riferimenti per seed 47/72: la copia
`run_configuration.frozen_best` conserva path/hash del YAML originale del seed
42, mentre il YAML locale cambia seed e hash. I `best_configuration` locali
hanno hash corretti; i YAML coincidono escludendo solo i due campi seed.
Non è una prova di corruzione: il riferimento ereditato dovrebbe indicare
esplicitamente la radice del seed di origine per evitare ambiguità nel replay.
Gli avvisi grezzi restano visibili in `audit.json` con questa interpretazione.

Inferenza CPU da checkpoint: seed 47 e 72 riproducono tutte le etichette e le
metriche storiche; sul seed 42 l'accuracy è 92,31198% invece di 92,29805%,
un campione corretto netto in più, e macro-F1 0,899022 invece di 0,898925.
La differenza massima delle probabilità è 0,004086 / 0,002381 / 0,002405.
Le baseline coincidono per tutti i seed. Non si può dichiarare identità numerica
fra backend CPU/GPU, né un replay del riaddestramento.

Il codice della run ha hash
`c142b62a35dc034b5e7115dcc25325b5d16650fa7ed8e965f6434edafe7616a1`;
quello attuale verificato nella precedente analisi ha hash
`6e218a0a39d71c351468f95dafdb9969a67d4893eca6b9b9a4f020862ff5d076`.
I 41 test e 10/10 guasti rilevati precedentemente non sono evidenza associabile
automaticamente a questa nuova run. Il digest di `qwen3.8` resta assente.

## Governance e allineamento alla presentazione

Tutti i seed: `technical_complete=true`, `acceptance_ready=false`,
`trl7_evidence_complete=false`, status `completed_with_warning`.
Il requisito delle ripetizioni con configurazione congelata ora passa nei
report finali. I report intermedi pending precedono la chiusura del summary.

Restano falliti: web retrieval, ablation della rappresentazione, mitigazioni
approvate. Restano pending: fault detection sul codice esatto, replay completo,
dimostrazione e sign-off indipendenti. La letteratura si riduce a tre fonti
`curated_excerpt` con quattro idee: questa run usa estratti integrati, non le
sei fonti recuperate dal web nella run precedente. Tracciabilità non equivale
a supporto scientifico delle specifiche inferenze o degli iperparametri.

Il Reviewer registra 5 / 6 / 2 eventi `review_unresolved` e segnala il fallimento
OOD del seed 47, ma la pipeline completa il training e reporting con warning.
Nella presentazione va distinto il completamento operativo dall'accettazione;
non sono ancora sostenibili protezione OOD stabile, replay completo o TRL 7.

## Priorità successive

1. Trattare il detector OOD del seed 47 come non idoneo nelle corruzioni medie/forti.
   Definire un criterio di idoneità su validation prima di qualunque test e
   prevedere l'esito esplicito «nessuno score idoneo».
2. Studiare score/calibration su un pannello di corruzioni comune ai seed,
   pubblicando AUROC minimo e accettazione OOD alla soglia scelta.
3. Eseguire ablation a configurazione congelata; non cambiare altri fattori.
4. Aggiungere un controllo ResNet comparabile e ricerca convenzionale a stesso
   spazio e budget per isolare il valore degli agenti.
5. Conservare corpus bibliografico verificato e modello Ollama con digest;
   generare test/fault evidence sul codice esatto ed effettuare replay storico.
6. Chiudere mitigazioni e dimostrazione indipendente prima dell'accettazione finale.

## Evidenze riproducibili

- `audit.json`: checksum e ricalcolo delle metriche storiche.
- `checkpoint_inference.json`: inferenza CPU e confronto con output storici.
- `supplemental.json`: invarianti fra seed, classi e accettazione delle corruzioni.
- `analyze.py`: analisi supplementare ripetibile, senza training.
- `baseline_predictions_seed_*.npz`: predizioni baseline ricalcolate dall'audit.

Non sono stati eseguiti nuovi training né riscritti i risultati della run.
