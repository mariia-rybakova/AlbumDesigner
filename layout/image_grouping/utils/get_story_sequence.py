import pandas as pd


KEYWORDS_PATH = 'i:\\Projects\\pic_time\\info\\keywords\\vendor_keywords.xlsx'


def get_tag_list(vendor=True, keywords_path=KEYWORDS_PATH):
    keywords_df = pd.read_excel(keywords_path, header=0)
    if vendor:
        vendor_df = keywords_df.drop('keyword', axis=1)
        keywords_df = keywords_df[~vendor_df.isnull().all(axis=1)]
    tags = list(keywords_df['keyword'])
    # tags = [tag.lower() for tag in tags]
    return tags


if __name__ == '__main__':
    tag_list = get_tag_list(vendor=True)
    print(tag_list)
    print(len(tag_list))