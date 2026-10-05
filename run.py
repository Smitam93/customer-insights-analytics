"""Synthetic Voice of Customer NLP and separate digital funnel analysis."""
from pathlib import Path
import json,sqlite3
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF
from sklearn.metrics import classification_report,f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
SEED=20261004
# Each held-out phrase family is excluded from training; labels never enter features.
TOPICS={
 'payments': ['The payment transfer {state}', 'My payment transaction {state}', 'The money payment {state}', 'A payment through the service {state}'],
 'login': ['The login password reset {state}', 'My login to the account {state}', 'The account login screen {state}', 'Signing in with my login {state}'],
 'fees': ['The service fee charge {state}', 'My fee on the statement {state}', 'The fee quoted for service {state}', 'Paying this service fee {state}'],
 'support': ['The support chat conversation {state}', 'My interaction with support {state}', 'The support team response {state}', 'Contacting customer support {state}'],
 'applications': ['The loan application process {state}', 'My credit application status {state}', 'The application documents step {state}', 'Submitting a loan application {state}']}
STATES={'negative':['failed and was frustrating','did not work and disappointed me','was delayed and very confusing','was terrible and caused trouble'],
        'neutral':['needs an explanation','is something I would like information about','has a question I want answered','requires clarification from someone'],
        'positive':['worked smoothly and was helpful','was quick and excellent','was easy and made me happy','worked well and satisfied me']}

def generate(seed=SEED,n=1800):
    rng=np.random.default_rng(seed); rows=[]
    for i in range(n):
        topic=rng.choice(list(TOPICS)); sentiment=rng.choice(list(STATES),p=[.5,.25,.25]); family=int(rng.integers(4))
        state=STATES[sentiment][family]
        text=TOPICS[topic][family].format(state=state)
        text=rng.choice(['Recently, ','Today, ','In my experience, '])+text+rng.choice(['. Please review.','. Thank you.','.'])
        rows.append({'feedback_id':f'F{i:05d}','template_family':family,
                     'channel':rng.choice(['chat','review','social']), 'text':text,
                     'topic_label':topic,'sentiment_label':sentiment,
                     'split':'test' if family==3 else 'train'})
    feedback=pd.DataFrame(rows)
    sessions=[]
    for i in range(6000):
        channel=rng.choice(['search','social','email','direct']);mobile=int(rng.binomial(1,.65))
        product=int(rng.binomial(1,{'search':.72,'social':.55,'email':.8,'direct':.67}[channel]))
        cart=product*int(rng.binomial(1,.30-.05*mobile));purchase=cart*int(rng.binomial(1,.6-.12*mobile))
        sessions.append({'session_id':f'S{i:05d}','channel':channel,'is_mobile':mobile,
                         'product_view':product,'add_to_cart':cart,'purchase':purchase,
                         'revenue_usd':purchase*float(rng.choice([35,60,90]))})
    return feedback,pd.DataFrame(sessions)

def model():
    return Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2)),
                     ('classifier',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=SEED))])

def main():
    feedback,sessions=generate(); feedback.to_csv(ROOT/'data/feedback.csv',index=False); sessions.to_csv(ROOT/'data/sessions.csv',index=False)
    train=feedback[feedback.split=='train'];test=feedback[feedback.split=='test'];scores={}; reports={};predictions=test[['feedback_id','text','channel','topic_label','sentiment_label']].copy()
    for label in ['topic','sentiment']:
        fitted=model().fit(train.text,train[label+'_label']);pred=fitted.predict(test.text)
        scores[label+'_macro_f1']=float(f1_score(test[label+'_label'],pred,average='macro'))
        majority=train[label+'_label'].mode().iloc[0]
        scores[label+'_majority_baseline_macro_f1']=float(f1_score(test[label+'_label'],[majority]*len(test),average='macro'))
        reports[label]=classification_report(test[label+'_label'],pred,output_dict=True,zero_division=0)
        predictions['predicted_'+label]=pred;predictions[label+'_confidence']=fitted.predict_proba(test.text).max(axis=1)
    # Explore topics using training text only, independently of supervised labels.
    vector=TfidfVectorizer(stop_words='english',min_df=3);x=vector.fit_transform(train.text)
    nmf=NMF(n_components=5,random_state=SEED,max_iter=600,init='nndsvda').fit(x)
    terms=np.array(vector.get_feature_names_out());topics={f'cluster_{i}':terms[c.argsort()[-8:][::-1]].tolist() for i,c in enumerate(nmf.components_)}
    predictions['needs_review']=(predictions.topic_confidence<.65)|(predictions.sentiment_confidence<.65)
    predictions['priority'] = np.where((predictions.predicted_sentiment=='negative')&(~predictions.needs_review),'triage','review')
    predictions.to_csv(ROOT/'outputs/feedback_predictions.csv',index=False)
    errors=predictions[(predictions.predicted_topic!=predictions.topic_label)|(predictions.predicted_sentiment!=predictions.sentiment_label)]
    errors.to_csv(ROOT/'outputs/error_audit.csv',index=False)
    # Dashboard uses test-only predictions; in-sample scores are never presented as held-out accuracy.
    dashboard=predictions.groupby('predicted_topic').agg(feedback_count=('feedback_id','size'),
                                                       negative_count=('predicted_sentiment',lambda v:(v=='negative').sum()),
                                                       review_count=('needs_review','sum')).reset_index()
    dashboard['negative_share']=dashboard.negative_count/dashboard.feedback_count
    dashboard.to_csv(ROOT/'outputs/voc_dashboard.csv',index=False)
    conn=sqlite3.connect(ROOT/'outputs/analytics.sqlite');sessions.to_sql('sessions',conn,if_exists='replace',index=False)
    funnel=pd.read_sql_query((ROOT/'sql/01_funnel_metrics.sql').read_text(),conn);funnel.to_csv(ROOT/'outputs/funnel_metrics.csv',index=False);conn.close()
    (ROOT/'outputs/topic_words.json').write_text(json.dumps(topics,indent=2));(ROOT/'outputs/classification_reports.json').write_text(json.dumps(reports,indent=2))
    result={'seed':SEED,'feedback_rows':len(feedback),'test_rows':len(test),'session_rows':len(sessions),**scores,
            'review_share':float(predictions.needs_review.mean()),'scope':'Synthetic templated text only; not real-world NLP accuracy',
            'limits':['Held-out phrase family prevents exact sentence-template duplication; vocabulary remains simple.',
                      'No multilingual slang, sarcasm, misspellings, multi-issue feedback or personally identifiable information.',
                      'NMF clusters are exploratory word groups and may reflect sentiment rather than business topics.',
                      'Classifier probabilities are uncalibrated; 0.65 review threshold is illustrative.',
                      'Feedback and website sessions are separate synthetic populations; no identity join is justified.',
                      'Funnel comparisons are descriptive; they do not establish channel or device causality.',
                      'NPS is not inferred from sentiment; actual survey responses would be required.']}
    (ROOT/'outputs/summary.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].bar(dashboard.predicted_topic,dashboard.negative_share,color='#126d7b');axes[0].set(ylabel='Predicted negative share',title='Held-out synthetic feedback');axes[0].tick_params(axis='x',rotation=25)
    pivot=funnel.pivot(index='channel',columns='is_mobile',values='purchase_rate');pivot.rename(columns={0:'Desktop',1:'Mobile'}).plot.bar(ax=axes[1],rot=0)
    axes[1].set(ylabel='Session purchase rate',title='Separate synthetic digital funnel');fig.savefig(ROOT/'outputs/customer_insights.png',dpi=160);plt.close(fig)
    (ROOT/'outputs/decision_memo.md').write_text(f'''# Synthetic customer-insights decision memo

On the held-out template family, topic macro F1 is {scores['topic_macro_f1']:.3f} and sentiment macro F1 is {scores['sentiment_macro_f1']:.3f}. These numbers verify the pipeline on controlled synthetic language; they are not realistic deployment benchmarks. Even perfect scores on this dataset would not validate a production model. At the illustrative confidence threshold, {result['review_share']:.1%} of feedback requires human review: this baseline is not ready for automated routing. Inspect `error_audit.csv` to diagnose errors against known synthetic labels.

Use the test-only feedback dashboard to propose a manual audit of frequent negative topics. Route low-confidence cases to reviewers; do not auto-resolve complaints. Review thresholds and probabilities require calibration on a genuinely varied, labeled corpus.

Website funnel metrics show where simulated sessions drop from product view to cart to purchase. Compare mobile versus desktop as a diagnostic and propose a usability experiment, rather than attributing differences causally to device or channel. Feedback and digital sessions are separate populations.

Next: add de-identified or appropriately licensed public language, group splits by author and near-duplicate phrase, temporal drift checks, a labeling rubric, error slices, calibrated abstention and human review. Never infer NPS or actual business improvement from sentiment classification.
''')
    print('Customer insights complete:',len(feedback),'synthetic texts and',len(sessions),'sessions')
if __name__=='__main__':main()
