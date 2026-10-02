# Revisione del codice e maggiore autonomia degli agenti

Le modifiche richieste sono implementate e verificate. La ricerca può scegliere
il numero di epoche e la politica di early stopping insieme agli altri
hyperparameter. Le configurazioni selezionate arrivano al trainer, al replay
dei seed e all'esportazione LightningCLI senza sostituzioni euristiche della
pazienza o delle epoche.

## Libertà di configurazione

| Parametro | Nuovo intervallo |
| --- | --- |
| Epoche proposte | 1–1.000 |
| Pazienza early stopping | 0–500; 0 lo disabilita |
| Metrica early stopping | `val_accuracy`, `val_macro_f1`, `val_loss` |
| `min_delta` | 0–1 |
| Batch size | Qualsiasi intero fra 4 e 4.096 |
| Larghezza | 8–1.024, multipli di 8; ResNet18 fino a 256, residual CNN fino a 512 |
| Profondità | 1–48; tiny CNN fino a 4, topologia ResNet18 fissa |
| Limite dei canali CNN | 8–4.096, multipli di 8 |
| Learning rate | 1e-7–1 |
| Weight decay / dropout / label smoothing | 0–1 / 0–0,95 / 0–0,5 |
| Gradient clipping | 0–100 |
| Patch ViT | 1, 2, 4, 7, 14, 28 |
| Teste / rapporto MLP / strati tokenizer CCT | 1–32 / 1–16 / 1–5 |
| Ricerca | Fino a 256 trial e 16 round, massimo 16 proposte per chiamata |
| Augmentation | Fino a 8 varianti distinte approvate |

Sono selezionabili anche momentum e Nesterov per SGD, beta ed epsilon per Adam,
learning rate minimo del cosine scheduler, frazione di warmup di OneCycle,
fattore e pazienza di ReduceLROnPlateau. Restano i vincoli di compatibilità:
le teste devono dividere la dimensione degli embedding, Nesterov richiede
momentum positivo e il learning rate minimo del cosine scheduler non può
superare quello iniziale. Le opzioni inattive vengono canonicalizzate.

`--max-epochs` è un tetto di risorse, con **default aumentato da 15 a 100** e
massimo 1.000. Il candidato sceglie il proprio orizzonte entro quel tetto.
`--search-epochs` limita i round esplorativi; il round finale usa il tetto
comune `--max-epochs`. Il numero effettivo è il minimo fra epoche proposte e
tetto del round, eventualmente ridotto dall'early stopping scelto dall'agente.
Una pazienza superiore alle epoche disponibili viene preservata.

## Ricerca e confronto dei candidati

La copertura obbligatoria delle architetture occupa al massimo uno slot per
round esplorativo. Con 8 trial e 3 round, l'allocazione normale è 3/3/2:
quattro proposte adattive, una prova di copertura, una promozione intermedia
e due finalisti. La copertura di tutte le cinque famiglie non è garantita con
un budget piccolo; le famiglie mancanti rimangono dichiarate nel report.

Le promozioni confrontano solo il gruppo con il più recente budget riuscito.
Il confronto finale usa lo stesso tetto di risorse, consentendo a ogni candidato
di scegliere un orizzonte più breve e una politica di arresto diversa. La
durata di esecuzione non determina più gli spareggi: si usa un ordinamento
stabile. I round senza spazio per proposte adattive non chiamano l'LLM.
Le richieste di arresto anticipato della ricerca non saltano il confronto finale.

Le decisioni successive ricevono epoche richieste, tetto assegnato, epoche
effettive, epoche completate e la parte finale delle curve di addestramento,
oltre alle metriche di validation. Il test set resta escluso dalle decisioni.
Se tutti i trial con il budget finale falliscono, la selezione fallisce
esplicitamente anziché esportare una configurazione finale non verificata.

## Correzioni OOD, artefatti e riproducibilità

- L'idoneità del detector viene stabilita sulla validation: ogni scenario deve
  raggiungere AUROC almeno 0,65 e accettare automaticamente al massimo il 20%
  dei campioni corrotti alla soglia calibrata sui campioni puliti. I limiti sono
  configurabili con `--ood-min-auroc` e `--ood-max-false-accept` e sono valori
  ingegneristici iniziali. Vanno fissati prima della valutazione sul test.
- Se nessun metodo è idoneo, il report registra `no_eligible_detector`, la
  copertura automatica è zero e tutti i casi vanno alla revisione. Il metodo
  diagnostico e i suoi tassi di falsa accettazione restano visibili. Un fallimento
  sul test viene segnalato senza modificare la scelta fatta sulla validation.
- Il rumore usa un seed separato dall'addestramento, default 1729. La verifica
  riguarda corruzioni gaussiane controllate e non dimostra robustezza OOD generale.
- Il file Lightning esportato condivide con il trainer monitoraggio del miglior
  checkpoint ed early stopping. Il checkpoint resta selezionato per
  `val_accuracy`; la metrica di arresto può essere scelta indipendentemente.
- L'integrità controlla ricorsivamente i riferimenti annidati, comprese ablation
  e snapshot bibliografici. Le nuove run congelate conservano una copia locale
  del file sorgente; i riferimenti storici si risolvono nel seed originario.
- L'identità del codice usa `canonical-text-v2`: percorsi relativi POSIX,
  terminatori di riga normalizzati e separazione esplicita dei contenuti.
  I checksum degli artefatti continuano a verificare i byte esatti.
  I manifest storici mantengono l'algoritmo precedente.

## Verifiche completate

[validation.json](validation.json) registra **50 test passati** e **10 guasti
simulati rilevati su 10**. Include la firma del codice verificato e la firma
del documento di evidenza. I test comprendono:

- instanziazione reale di LightningCLI dal payload esportato;
- addestramento CPU con early stopping su `val_loss`, `min_delta` e momentum
  scelti dall'agente;
- preservazione di un orizzonte breve e di una pazienza lunga nella configurazione
  selezionata;
- ricerca con proposte adattive e finalisti confrontati al budget comune;
- addestramento e replay con predizioni identiche, anche per un seed congelato;
- rifiuto OOD deciso sulla validation, corruzioni accettate e invio alla revisione;
- manomissione di evidenze annidate e compatibilità delle firme del codice.

[historical_compatibility.json](historical_compatibility.json) documenta la
riapertura tipizzata e il controllo d'integrità dell'ultima run
`pathmnist_20261002T104242464569Z`: 58 artefatti per seed 42, 48 per seed 47 e
48 per seed 72, **nessun problema d'integrità**. Il controllo è stato effettuato
in sola lettura. Le metriche di quella run non sono state ricalcolate con il
nuovo codice e il suo replay richiede ancora la revisione originale.

La verifica usa dati sintetici piccoli e CPU. Non è stato avviato un nuovo
addestramento completo PathMNIST; prestazioni e consumo di memoria dei nuovi
massimi devono essere misurati in una nuova run. Configurazioni più grandi
possono esaurire la memoria: il trial fallito viene registrato, senza ridimensionare
silenziosamente il modello.

## Esempio di nuova run con maggiore spazio di ricerca

Dalla cartella `v2`, con il dataset e Ollama disponibili:

```bash
python run.py --seeds 42,47,72 --ablation-suite \
  --search-trials 24 --search-rounds 4 --search-epochs 10 --max-epochs 300 \
  --ollama-base http://localhost:11434 --ollama-model qwen2.5:7b --require-llm
```

Il comando consente agli agenti di scegliere fino a 300 epoche; non impone di
eseguirle tutte. Per usare anche la ricerca bibliografica online, aggiungere
`--research-online --require-research --literature-cache runs/literature_live_v3`
con una cache nuova o già composta da fonti recuperate online.
