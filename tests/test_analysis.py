import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import generate,model

class CustomerTests(unittest.TestCase):
    def test_template_family_and_text_split_is_disjoint(self):
        f,_=generate();train=f[f.split=='train'];test=f[f.split=='test']
        self.assertFalse(set(train.template_family)&set(test.template_family))
        self.assertFalse(set(train.text)&set(test.text))
    def test_nested_session_stages(self):
        _,s=generate()
        self.assertEqual(s.session_id.nunique(),len(s))
        self.assertTrue((s.purchase<=s.add_to_cart).all())
        self.assertTrue((s.add_to_cart<=s.product_view).all())
        self.assertEqual(float(s.loc[s.purchase==0,'revenue_usd'].sum()),0.)
    def test_unseen_vocabulary_is_not_fitted(self):
        f,_=generate();train=f[f.split=='train']
        fitted=model().fit(train.text,train.topic_label)
        self.assertNotIn('xylophoneterm',fitted['tfidf'].vocabulary_)
        result=fitted.predict(pd.Series(['xylophoneterm payment']))
        self.assertEqual(len(result),1)
        self.assertNotIn('xylophoneterm',fitted['tfidf'].vocabulary_)
if __name__=='__main__':unittest.main()
