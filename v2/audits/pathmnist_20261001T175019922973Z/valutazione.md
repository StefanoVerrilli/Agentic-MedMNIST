# Audit della run PathMNIST del 1 ottobre 2026

Audit eseguito il 2 ottobre 2026 dopo il caricamento delle evidenze binarie.
La run storica non è stata modificata dall'audit. I risultati seguenti sono
verifiche successive, conservate separatamente.

## Esito

Le metriche registrate sono confermate. Il miglioramento medio rispetto alla
baseline fissa è 5,95 punti di accuracy. Non sono ancora dimostrati il contributo
isolato degli agenti, la riproducibilità completa del training o il TRL 7.

| Seed | Accuracy agentic | Accuracy baseline | Macro-F1 agentic | Macro-F1 baseline |
|---|---:|---:|---:|---:|
| 42 | 0,889554 | 0,821309 | 0,867082 | 0,750857 |
| 47 | 0,895125 | 0,840251 | 0,871257 | 0,784172 |
| 72 | 0,883426 | 0,828134 | 0,853015 | 0,757331 |

`audit.json` verifica checksum degli artefatti, transcript di reasoning e file
referenziati, consistenza tra archivi di predizione, accuracy, macro-F1,
balanced accuracy, soglia calibrata, coverage, accuracy selettiva, coda di
revisione, AUROC delle singole corruzioni e aggregazione. Su tutti i seed:
zero problemi d'integrità, zero conversioni newline residue e verifiche numeriche
superate. I file ora caricati risolvono le assenze della precedente ispezione.

`checkpoint_inference.json` contiene una verifica aggiuntiva: caricamento sicuro
dei checkpoint (`weights_only=True`), ricostruzione dei modelli e inferenza CPU
sull'intero test. Accuracy e macro-F1 di entrambi i modelli coincidono con i
report a sei decimali; tutte le etichette agentic coincidono con quelle storiche.
Le probabilità non sono numericamente identiche: differenza massima assoluta
0,001655 / 0,002573 / 0,004550 sui seed 42 / 47 / 72. Pertanto non è una prova
di replay esatto, né di riaddestramento riproducibile. Le predizioni baseline
ricalcolate sono conservate nei tre `baseline_predictions_seed_*.npz`.

## Confronto e reasoning

Il modello agentic ha 6.288.057 parametri, la baseline 5.433: oltre mille volte
meno parametri nel controllo. Inoltre differiscono augmentation e impostazioni
di training; il sistema agentic paga anche otto trial di ricerca per seed.
Il guadagno non può essere attribuito causalmente al reasoning.

La spiegazione storica che aumenta la profondità ResNet tramite `depth` è errata:
la topologia della ResNet18 è fissa, mentre `hidden` cambia la larghezza. Anche
l'affermazione di 2–4 milioni di parametri sottostima il modello finale.
I campi transformer comuni non trasformano una CNN in un modello ibrido.

Nel codice corrente era già presente la correzione del fingerprint ResNet che
ignora `depth`. L'audit ha aggiunto chiarimenti nei prompt di Search e Reviewer
su questi aspetti, sulle primitive integrate rispetto alle estensioni nuove,
sul supporto scientifico delle inferenze e sull'aggregazione OOD. Nessuna
spiegazione o decisione storica è stata riscritta.

## OOD e astensione

Sul seed 72, per rumore gaussiano sigma=0,4, i 7.180 campioni di test corrotti
sono tutti predetti come background. La confidenza max-softmax media è 0,967154,
contro 0,917280 sul test pulito; l'AUROC è 0,133864. Il fallimento del detector
è riproducibile dagli array, non un errore di arrotondamento del report.
Questa evidenza mostra overconfidence sotto quella corruzione; non stabilisce
da sola la causa nei pesi o nella pipeline.

L'aggregato del seed 42 (0,823189) è la media degli AUROC max-softmax delle tre
severità. La diversa media di tutte e sei le righe include anche l'entropia:
il rilievo storico sulla presunta aggregazione pooled non era fondato.

La coverage test è 62,24% / 65,38% / 58,90%, contro il target validation dell'80%.
L'accuracy sui casi accettati è 96,40% / 96,10% / 97,00%, con 2.711 / 2.486 /
2.951 casi inviati alla coda. Questi dati non dimostrano una revisione umana
conclusa. Il recall dello stroma associato al tumore resta 49,4–55,1%.

## Validazione del codice corrente

`current_source_validation.json` è un documento checksummed: 41 test passati e
10/10 fault injection rilevate. Il risultato riguarda i dieci guasti modellati,
non tutte le possibili inconsistenze o allucinazioni. Il test di replay sintetico
passa, ma non è il replay del training della run storica.

Hash del codice corrente verificato:
`6e218a0a39d71c351468f95dafdb9969a67d4893eca6b9b9a4f020862ff5d076`.
Hash registrato dalla run storica:
`01a825dee594ac232140dbbe6d53578f01a3ac4d0ecb4d5ed8a9e7a063105076`.
Gli ambienti differiscono (Windows/Python 3.14/CPU contro Linux/Python 3.11/GPU).
Le nuove evidenze non devono essere associate al vecchio hash per far passare
l'accettazione storica.

## Rilievi ancora aperti e presentazione

- I seed storici ripetono la ricerca con configurazioni diverse, non il solo training.
- Mancano ablation della rappresentazione e confronti a spazio/budget comparabili.
- Il digest del modello Ollama storico non è registrato.
- Le citazioni risolvibili e le quote verificabili non provano ogni inferenza scientifica.
- Replay completo, mitigazioni approvate e dimostrazione/sign-off indipendenti restano aperti.

La presentazione può riportare il completamento del MVP e il confronto con la
baseline fissa. Il KPI di fault detection è ora misurato sul codice corrente,
con scope e hash dichiarati. Riproducibilità completa e TRL 7 restano target.

La sequenza eseguibile e i controlli ancora da implementare sono specificati in
`v2/docs/protocollo_validazione_pathmnist.md`. Nuovi training completi non sono
stati avviati: questo audit ha eseguito inferenza dai checkpoint e i piccoli
training previsti dai test, su una macchina senza GPU CUDA disponibile.
