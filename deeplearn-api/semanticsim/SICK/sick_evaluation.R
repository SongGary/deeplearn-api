ifile = 'stslstm_trial_result00.txt';
gold = 'SICK_new_trial.txt';
ifile = 'stslstm_test_result00.txt';
gold = 'SICK_test_annotated.txt';

read.delim(ifile, sep="\t", header=T, stringsAsFactors=F) -> score;
read.delim(gold, sep="\t", header=T) -> gold;

score <- score[order(score$pair_ID), ];
gold <- gold[order(gold$pair_ID), ];



if (FALSE %in% (score$pair_ID==gold$pair_ID)){
	print("ERROR: pair IDs in score set and gold set do not correspond")
	quit()
	}


print(paste("Processing ", ifile, sep=""));

if (TRUE %in% is.na(score$entailment_judgment)){
	print("No data for the entailment task: evaluation on relatedness only")
	}else if (FALSE %in% (unique(score$entailment_judgment) %in% unique(gold$entailment_judgment))){
	print("ERROR: wrong level(s) in entailment judgments")
	}else{
	accuracy <- sum(score$entailment_judgment == gold$entailment_judgment) / length(score$entailment_judgment)*100
	print(paste(paste("Entailment: accuracy ", accuracy, sep=""),"%",sep=""))
	}
	
if (TRUE %in% is.na(score$relatedness_score)){
	print("No data for the relatedness task: evaluation on entailment only ")
	}else if (is.numeric(score$relatedness_score)==FALSE){
	print("ERROR: wrong format for relatedness scores")
	}else{
	pearson <- cor(score$relatedness_score, gold$relatedness_score)
	#print(paste("Relatedness: Pearson correlation ", pearson, sep=""))
	spearman <- cor(score$relatedness_score, gold$relatedness_score, method = "spearman")
	#print(paste("Relatedness: Spearman correlation ", spearman, sep=""))
	MSE <- sum((score$relatedness_score - gold$relatedness_score)^2) / length(score$relatedness_score)
	#print(paste("Relatedness: MSE ", MSE, sep=""))
	print(pearson)
	print(spearman)
	print(MSE)
	}