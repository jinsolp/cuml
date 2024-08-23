#=============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

import jsonlines
from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('all-mpnet-base-v2')

sentences = []
cnt = 0
with jsonlines.open('data/Electronics.json') as reader:
    for obj in reader:
        cnt += 1
        if cnt % 1000000 == 0:
            print(f"{cnt}")
        text = obj.get('reviewText', None)
        if text is not None:
            sentences.append(text)

embeddings = model.encode(sentences, show_progress_bar=True)

print(embeddings.shape)
 
with open('data/amazon-elec.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
