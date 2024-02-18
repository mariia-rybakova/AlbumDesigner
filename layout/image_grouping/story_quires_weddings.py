import nltk
nltk.download('punkt')

from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

stemmer = Stemmer("english")
summarizer_lsa = Summarizer(stemmer)
SENTENCES_COUNT = 10
summarizer_lsa.length_limit = 7

def generate_summary(text):
    # Create a plaintext parser
    parser = PlaintextParser.from_string(text, Tokenizer('english'))

    summary = summarizer_lsa(parser.document, SENTENCES_COUNT)
    summary_list = [str(sentence) for sentence in summary]
    return summary_list

# Different Example stories
beach_wedding_story = "On the sandy shores, groom, adorned in beachy attire, prepared for the wedding. The bride, surrounded by friends, did the same. As waves gently crashed in the background, groom and bride walked down the sandy aisle. The ceremony, with the sound of the ocean as groom's and bride's soundtrack, saw groom and bride exchanging rings and kisses. Post-ceremony, a photosession amidst the sun-kissed beach captured the love shared with friends and family. With the setting sun, groom and bride moved to a beachside venue for a night of dancing. The laughter echoed along with the waves, concluding groom's and bride's beach wedding with a happily-ever-after."

indoor_venue_wedding_story = "Within the grandeur of an indoor venue, the groom, donned in a classic suit, prepared for the wedding. The bride, surrounded by friends, did the same. grooms and bride walked down the elegantly decorated aisle as the ceremony unfolded. The exchange of rings and kisses marked the beginning of groom and bride married life. In the opulent venue, a sophisticated photosession followed, capturing moments with family and friends. The night continued with a dance, filling the venue with joy and celebration, culminating in a blissful ending."

backyard_garden_wedding_story = "In the cozy ambiance of a backyard garden, the  groom, dressed in a charming ensemble, prepared for the wedding. Surrounded by friends, the bride also got ready. bride and groom strolled through a flower-strewn aisle, where the intimate ceremony began. Amidst the blooming flowers, groom and bride exchanged rings and kisses, sealing their commitment. The garden became the backdrop for a heartfelt photosession with family and friends. As evening descended, the backyard transformed into a dance floor. Laughter and love filled the air, providing the perfect ending to groom's and bride's magical garden wedding."

gays_wedding_story = "In the celebration of a wedding for two gay guys, their story began with a sweet kiss. The first groom prepared alongside the second, surrounded by their friends. Walking down the aisle, the ceremony commenced, with their beloved ones beaming with happiness.Exchanging rings and sealing their commitment with a kiss, the two grooms embarked on a photosession, capturing cherished moments with friends and family. As night fell, dancing ensued, and the joyous celebration continued. Together, two grooms found happiness in each other\'s company, marking the start of a beautiful journey as a married couple."

desert_wedding_story = "In a discreet location, the groom eagerly awaited his bride. She playfully approached him from behind, gently poking him. As he turned, a radiant smile lit up his face at the sight of her. Amidst the sunlight filtering through large stone breaks, groom and bride shared a kiss, standing in bliss.Capturing moments with poses, kisses, and intertwined hands, groom and bride created memories before heading to their ceremony. Approaching the officiant, the intimate ceremony began with just the bride, groom, and officiant. groom and bride exchanged rings and sealed their vows with a heartfelt kiss.After the ceremony, surrounded by family and friends, groom and bride took joyous pictures. Alone, groom and bride walked to secluded spots between two massive stone breaks, engaging in more photosessions. With each kiss and gaze into each other\'s eyes, groom and bride celebrated the unique bond groom and bride had formed, cherishing the start of their shared journey."

normal_wedding_story = "On their wedding day, the groom excitedly prepared for the ceremony. In the second room, as he turned around, a wave of happiness hit him upon seeing his bride in her elegant white dress. Approaching her, he couldn't resist kissing her, and groom and bride held hands, capturing joyful poses before the ceremony.Walking hand in hand to the officiant, groom and bride stood facing each other, exchanging vows and rings. The officiant pronounced them married, sealing it with a kiss. groom and bride ventured into a garden street for a whimsical photoshoot, creating memories with peculiar effects.Afterwards, at a nearby bar, groom and bride shared drinks and food, sealing their love with a kiss. Holding hands, groom and bride stepped into the journey of marriage, ready to embrace their shared future."

# Extract sequence points from each story
print("Summaization of the stories")
beach_wedding_sequence = generate_summary(beach_wedding_story)
indoor_venue_sequence = generate_summary(indoor_venue_wedding_story)
backyard_garden_sequence = generate_summary(backyard_garden_wedding_story)
gay_sequence = generate_summary(gays_wedding_story)
desert_sequence = generate_summary(desert_wedding_story)
normal_sequence = generate_summary(normal_wedding_story)

# Print the results
print("Beach Wedding Sequence Points:")
print(beach_wedding_sequence)

print("\nIndoor Venue Wedding Sequence Points:")
print(indoor_venue_sequence)

print("\nBackyard Garden Wedding Sequence Points:")
print(backyard_garden_sequence)


print("\nGay Wedding Sequence Points:")
print(gay_sequence)

print("\nDesert Wedding Sequence Points:")
print(desert_sequence)


print("\nNormal Wedding Sequence Points:")
print(normal_sequence)
