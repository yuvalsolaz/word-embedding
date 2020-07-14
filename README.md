# word-embedding
word embedding: load pretrained, use for words calculations 

git clone https://github.com/yuvalsolaz/word-embedding.git
cd word-embedding/src/

Download Hebrew words vectors:
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.he.300.vec.gz 
gzip -d cc.he.300.vec.gz

More language vectors can be found here:
https://fasttext.cc/docs/en/crawl-vectors.html

Find top 10 closest words :
python wordEmbeddingEngine.py  
usage: python wordEmbeddingEngine.py <vector_file_name> <max-words> <word1> <word2>

Example:
python wordEmbeddingEngine.py  cc.he.300.vec 2000 קרוב
top match:

קרוב ==> להגיע		norm=0.9243235093840252	cosine=0.5266592305831949
קרוב ==> בסמוך		norm=0.9587658525416933	cosine=0.48556666654460867
קרוב ==> יחסית		norm=0.9741657713141024	cosine=0.43273040645506367
קרוב ==> אפילו		norm=0.9923271587536039	cosine=0.3525532535381924
קרוב ==> לפחות		norm=0.9933543728196902	cosine=0.36522368563577334
קרוב ==> בניגוד		norm=0.9982551677802626	cosine=0.37468829113307117
קרוב ==> בנוסף		norm=1.0030295160163534	cosine=0.3287620686345813
קרוב ==> מתאים		norm=1.0032754806133757	cosine=0.40444865878270925
קרוב ==> קרובות		norm=1.0033236267526047	cosine=0.3521064058099679
קרוב ==> מספיק		norm=1.0041790378214435	cosine=0.38462715046758306

References

T. Mikolov, E. Grave, P. Bojanowski, C. Puhrsch, A. Joulin. Advances in Pre-Training Distributed Word Representations

@inproceedings{mikolov2018advances,
  title={Advances in Pre-Training Distributed Word Representations},
  author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
