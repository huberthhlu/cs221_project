import re
import string
import emoji
from zhon import hanzi

class Filter_Text:

    def filter_emoji(fun):
        def wrapper(self, text=''):
            result_text = fun(self, text)
            return re.sub('(@@.*@@)', '', emoji.demojize(
                '{0}'.format(result_text), 
                delimiters=('@@', '@@')
            ))

        return wrapper

    # def filter_punctuation(fun):
    #     def wrapper(self, text=''):
    #         # punc = [u'「', u'」', u'（', u'）',u'(',u')',u'[',u']',u'{',u'}',u'"',u"'"]
    #         result_text = fun(self, text)
    #         return ''.join([
    #             w for w in result_text 
    #                 if (
    #                     w not in string.punctuation and
    #                     w not in hanzi.punctuation and 
    #                     w not in punc
    #                 )
    #         ])
    #     return wrapper

    @filter_emoji
    # @filter_punctuation
    def filtet_text(self, text=''):
        return text