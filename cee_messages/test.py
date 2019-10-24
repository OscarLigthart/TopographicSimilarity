# open a file and look at the contents

import pickle

path = '1/messages/message_from_0_at_500001'

data = pickle.load(open(path, 'rb'))

print(len(data))