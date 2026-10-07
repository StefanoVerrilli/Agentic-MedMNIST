# Agentic MedMNIST v3

V3 include Multi-Scale Transformer, Feature Pyramid Transformer e agenti che
creano codice Python relativo alla singola run. V1 e v2 restano indipendenti.

## Avvio autonomo locale

Dalla directory `v3`, con l'ambiente Python attivo e Ollama disponibile:

```bash
python run.py \
  --execution-mode agent_autonomous \
  --device cuda \
  --ollama-base http://localhost:11434 \
  --ollama-model qwen2.5:7b
```

Usare `--device cpu` quando CUDA non è disponibile. Non servono worker remoto,
SSH o Docker. Il codice generato gira in subprocess locali con lo stesso
interprete e le stesse dipendenze della pipeline, su Windows o Linux.
Questo intervento predispone il codice; non avvia la pipeline.

Gli agenti scelgono tutti i parametri sperimentali attivi, le epoche iniziali,
l'early stopping, i nuovi trial, le continuazioni e la conclusione della ricerca.
Non sono accettati `--max-epochs`, `--search-epochs`, `--search-trials`,
`--search-rounds` o `--max-generated-bundles` nella modalità autonoma.
Non esiste un tetto fisso di 1000 epoche. Ollama è obbligatorio: decisioni
non valide interrompono la run dopo i tentativi previsti, senza fallback euristici.

Se una decisione viene rifiutata, Ollama riceve sia la risposta precedente sia gli
errori di validazione per correggerla. `decision_log.jsonl` registra la causa,
l'identificativo della richiesta e un estratto della risposta rifiutata.
`--llm-retries` assegna tentativi aggiuntivi separati agli errori di trasporto
e alle risposte non valide: un timeout non consuma la possibilità di correggere
il JSON. Con il default di 1 retry si effettuano al massimo 3 chiamate logiche
per decisione. La validazione resta obbligatoria, incluso `optimizer_eps`
nell'intervallo da `1e-12` a `0.01`.
Se l'unico errore è `experiment.training.optimizer_eps` fuori intervallo, il retry
chiede a Ollama soltanto quel valore, espresso come stringa in notazione scientifica
con uno schema che ne vincola l'intervallo. Il codice e gli altri parametri della
proposta vengono conservati; la decisione completa viene nuovamente validata.
Il log registra il valore rifiutato e la correzione mirata in `field_repairs`.
Lo schema distingue `new_trial`, `continue_trial` e `finish_search`: i campi
inattivi devono essere assenti o null. Per `new_trial` l'identificativo del trial
viene assegnato dal framework. Le correzioni successive ricevono la proposta più
recente e gli errori precisi, anche quando una correzione di epsilon rende visibile
un errore negli argomenti dell'azione. I tentativi rifiutati restano nel log anche
in caso di recupero riuscito. Esauriti i retry, la ricerca autonoma non viene
riavviata dall'orchestratore, evitando di perdere la storia o riusare i checkpoint.
Per generazioni lente si configura `--llm-timeout` (default 90 secondi). Se Ollama
segnala `done_reason=length`, si aumenta `--llm-num-predict` (default 16384 token,
anche tramite `AGENTIC_LLM_NUM_PREDICT`). Sono limiti delle chiamate LLM;
le epoche di training restano scelte dagli agenti.
Una risposta terminata con `done_reason=length` viene rifiutata anche se il JSON
risulta sintatticamente valido.

Il revisore della ricerca riceve `analysis` e `transformer_guidance` completi;
le anteprime abbreviate degli altri artefatti sono segnalate in `compacted_fields`.
Le richieste di revisione della ricerca includono il testo precedente e i rilievi.
`evidence_provenance` distingue abstract recuperati, estratti curati non verificati
e riferimenti esterni non verificati. Tutte le sette architetture integrate,
inclusi i transformer, sono già disponibili: l'approvazione riguarda soltanto
le nuove proposte di estensione e non impone una priorità artificiale alle CNN.

Split ufficiali, metriche, seed e protocollo di valutazione restano gestiti dal
framework. La ricerca vede solo training e validation; il test viene utilizzato
dopo il congelamento del vincitore. La baseline è omessa nella modalità autonoma.
I report dichiarano baseline e confronto non eseguiti, senza attestare i requisiti
che li richiedono. Per default si usa il dataset completo e il seed 42.

## Codice e artefatti della run

I sorgenti degli agenti sono versionati in `run/generated/<bundle>/vNNN` e
verificati tramite checksum. Ogni operazione crea una directory distinta in
`run/blobs/local_jobs/<uuid>` contenente input, copia del framework, copia del
bundle, output, directory di lavoro e `process.log`. Non si importano sorgenti
generati nel processo principale e non si modificano i moduli del repository
per integrare gli esperimenti.

Il subprocess è una separazione di esecuzione, non una sandbox: usa i permessi
normali dell'utente, senza restrizioni artificiali di rete, filesystem o memoria.
Non viene imposto un limite di RAM. `--operation-timeout` configura il timeout
per operazione, default 3600 secondi; `--worker-timeout` rimane un alias compatibile.
Un timeout interrompe il processo e i suoi figli e conserva il log del job.
Non è imposto un limite complessivo di durata della ricerca.

## Training adattivo e replay

### Rilascio GPU, verifica e arresto degli errori

Nelle run avviate da `run.py`, prima di ogni subprocess CUDA il controller
scarica soltanto il modello Ollama configurato, tramite `/api/generate` con
`keep_alive: 0`, e ne verifica l'assenza in `/api/ps`. Il servizio resta attivo;
il modello viene ricaricato alla richiesta LLM successiva. Tra decisioni e retry
consecutivi si conserva il `--keep-alive` configurato. Il passaggio ha una deadline
di 30 secondi: se non viene confermato, la run fallisce prima di avviare il worker.
CPU, offline e replay non effettuano questa operazione. Durante una run CUDA usare
la GPU esclusivamente per la pipeline: altri client possono ricaricare Ollama.

`decision_log.jsonl` registra inizio, fine ed errore delle decisioni, tentativi LLM,
tempo di caricamento, preparazione degli array, staging, PID, uscita del subprocess
e heartbeat ogni 15 secondi durante l'attesa. I job conservano `process.log`;
i worker falliti scrivono anche `output/failure.json`. Il helper adattivo stampa
le metriche e aggiorna `training.jsonl` al termine di ogni epoca.
Su Windows il worker viene avviato sospeso, associato a un Job Object e poi
riattivato: il timeout termina anche i discendenti senza dipendere da `taskkill`.
Su Linux viene terminato il gruppo del processo.

Gli input generati sono `float32` NCHW `[N,3,28,28]`, già trasformati secondo
la rappresentazione scelta; non vanno trasposti o normalizzati nuovamente.
Il prompt fornisce un esempio basato su `fit_model` e `batched_logits`, con
dataset in RAM e trasferimento sulla GPU dei soli batch. I loop custom restano
ammessi, ma il worker verifica il device durante l'esecuzione dei modelli.
La verifica autonoma prova training sintetico, ripresa da un percorso `resume`
in una nuova directory di output e inferenza dal checkpoint. Non modifica il
budget scientifico scelto dall'agente per il training reale.

Un errore `code_validation` richiede la riparazione dei sorgenti precedenti:
il prompt include traceback e contratto; la correzione crea una nuova versione
dello stesso bundle con riferimento al genitore. La ricerca termina dopo
3 trial falliti consecutivamente o 2 errori consecutivi con la stessa firma.
I contatori si azzerano soltanto dopo un trial completo riuscito. Storage,
invarianti del controller e rilascio GPU falliti interrompono immediatamente la
ricerca. L'orchestratore scrive lo stato terminale `failed:model_search` e il dossier.
I trial aggiungono `failed_operation`, `failure_kind` e `failure_signature`; i
vecchi artefatti restano leggibili. Il replay richiede sempre prompt e schemi
compatibili: le trascrizioni della versione precedente non sono riutilizzate
silenziosamente con i nuovi contratti.

Validazione sul server, dopo i test locali:

```bash
python run.py --execution-mode agent_autonomous --device cuda \
  --ollama-base http://localhost:11434 --ollama-model qwen3.8:latest \
  --llm-timeout 300 --quick --output-root runs/validation_quick
```

Controllare gli eventi di unload e i heartbeat, la VRAM libera prima del training,
e il completamento della pipeline. Solo dopo avviare una nuova run senza `--quick`.
Cache dataset, retention, deadline di preparazione e riduzione di `fsync` sono
rinviate; gli artefatti della run problematica vengono conservati.

Ogni segmento salva `model.ckpt`, il migliore modello per la valutazione, e
`resume.pt`, lo stato più recente per la continuazione: modello, optimizer,
scheduler, RNG, loader al confine di epoca, storia ed early stopping.
L'agente sceglie `new_trial`, `continue_trial` oppure `finish_search`.
Una continuazione mantiene codice e parametri; cambiamenti richiedono un nuovo trial.
Un trial già fermato dall'early stopping o con OneCycle esaurito richiede un nuovo
trial. Il periodo iniziale del cosine scheduler viene conservato nella ripresa.

Si confrontano anche trial con durate diverse: accuracy di validation, poi
macro-F1 nella tolleranza del protocollo. Il checkpoint vincente viene congelato
prima del test, senza retraining aggiuntivo. Le ripetizioni su altri seed
riutilizzano configurazione e sequenza dei segmenti; il replay usa le decisioni
archiviate senza contattare Ollama.

```bash
python run.py --replay-run runs/<esperimento>/seed_42 --output-root runs/replay
python generated_cli.py fit --config runs/<esperimento>/seed_42/best_config.yaml --run-root runs/<esperimento>/seed_42 --device cpu
```

## Architetture integrate

Il portfolio comprende sette famiglie: `tiny_cnn`, `residual_cnn`, `resnet18`,
`vision_transformer`, `compact_transformer`, `multi_scale_transformer` e
`feature_pyramid_transformer`. I modelli integrati si possono anche utilizzare
senza generare codice; il percorso `legacy` mantiene le opzioni di ricerca precedenti.

`multi_scale_transformer` fonde token di patch di dimensioni diverse in un
encoder comune. Default del portfolio: scale `[2,4,7]`, larghezza 64, profondità 2,
quattro teste, rapporto MLP 4, dropout 0,1 e pooling medio. Le scale ammesse sono
2, 4, 7 e 14, almeno due distinte; le codifiche posizionali sono separate per scala.

`feature_pyramid_transformer` implementa ST, GT e RT del
[paper ECCV 2020](https://arxiv.org/abs/2007.09451): ST usa prodotto scalare con
Mixture of Softmaxes a due componenti; GT distanza euclidea negativa con quattro
componenti; RT pesi di canale e convoluzioni 3×3 dalla scala fine alla grossolana.
Tutti i rami leggono la piramide originale prima di concatenazione e fusione 3×3.
Il backbone ResNet-18 compatto, senza pesi preaddestrati, ha base width 32 e livelli
28,14,7,4. Default: `hidden=256`, un solo blocco FPT, pooling dei quattro livelli,
concatenazione e testa a nove classi; AdamW, batch 32, dropout 0,1 sulla testa.
`depth` è canonicalizzato a 1; teste, patch, scale e posizioni sono inattivi.
La fedeltà riguarda gli operatori FPT: backbone, testa e training sono adattati
a PathMNIST, senza replicare gli esperimenti COCO.

## Interfaccia degli esperimenti

`experiment.py` espone `build_model(context)`, `train(context)` e `predict(context)`;
può includere moduli ausiliari e implementare modelli, preprocessing, loss,
optimizer e loop nuovi. Il modello riceve float NCHW 28×28 RGB e produce nove logits.
`context` contiene configurazione, parametri liberi, array dei dati, percorsi
assoluti `output` e `checkpoint`, target cumulativo `epochs`, `start_epoch`,
`segment_epochs` e `resume`. Usare questi percorsi, senza directory hardcoded.

Training riceve solo array train/val; inferenza riceve immagini senza target.
`train` restituisce storia cumulativa e riepilogo, con `stop_reason` pari a
`segment_complete` o `early_stopping`, salvando i checkpoint nella directory output.
`worker_runtime.fit_model` implementa il contratto adattivo; i loop personalizzati
devono conservare stato equivalente e gli identificativi della configurazione e dei dati.
L'export corrente usa `run-experiment-v1`; i metadati SSH/Docker dei vecchi export
vengono ignorati e gli esperimenti vengono eseguiti localmente. Un replay storico
richiede comunque la versione originale dei sorgenti registrata nel manifest.
