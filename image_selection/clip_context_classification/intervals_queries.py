category_queries = {
'bride_preparation_before_wedding': ['bride getting dress','bride getting hair done', 'bride getting makeup done','bride putting her jewelry', 'bride with her bridesmaid preparing for the wedding', 'bride with her perfume','bride\'s shoes','wedding dress only', "wedding dress hanging in the wall or something", "bride carrying her wedding dress","hairdresser is styling bride\'s hair","makeup artist putting makeup for the bride","bride and bridesmaids in pajamas","bride and bridesmaids in pajamas drinking glass of drink" ],
'groom_preparation_before_wedding': ['groom and groomsbestmen preparing before the wedding ', 'groom is getting dress his suit', 'groom putting his cufflinks ', 'groom with his groomsmen before wedding', 'groom\'s shoes', 'grooms and his groomsmens trying to put the tie of the suit'],
'wedding_reception_before_ceremony':['The bride\'s parents welcoming guests with smiles and warmth as they enter the venue.','Guests mingling in anticipation, exchanging greetings and well wishes','groom is welcoming his guests at the door of the hall', 'guests hugging and gretting grooms before the wedding starts'],
'couple_arrival_before_ceremony' :['groom or bride in a vehicle', 'bride and groom in limousine','bride and groom standing in front of limousine','bride getting outside the car','brides has arrived by car', 'bride has arrived to the venue\'s wedding'],
'wedding_ceremony' :['ceremony', 'wedding ceremony','bride is doing performance during the ceremony','bride or groom giving speech to each other during the ceremony', 'groom is doing performance during the ceremony', 'bride and groom doing special dance in the ceremony'],
'speech_toasts':['bridesmaids giving speech','one of the brides family giving speech', 'one of the grooms family or friends giving a speech','someone giving speech during the wedding', 'guests and bride and groom making toasts for the speech'],
'photo_session_bride_and_groom': ['bride and groom', 'bride and groom with beautiful background',
                        'A personal moment between the groom and bride',
                        ' intimate moment in a serene setting between bride and groom',
                        'A black and white image that adds variety to the photo shoot of bride and groom',
                        'Bride and groom in a different setting', 'Smiling bride and groom in the photo',
                        'groom and bride kissing each other', 'bride and groom are holding hands',
                        'bride and groom in a gorgeous standing ', 'bride and groom kissing each other in a romantic way'],
'bride_bridesmaids_photosession':['photo session brides with her maids after ceremony', 'bride and her bridesmaids holding flowers without showing faces','brides with her maids wearing dressings', 'bride and bridesmaids standing for group picture wearing dresses', 'bridesmaids doing party for bride after wedding ceremony wearing all dresses'],
'groom_bestmen_photosession':['groom with his best men standing together', 'group photo of groom with his bestmen', 'groom laughing with his friends bestmens'],
'dancing': ['bride and groom with bridesmaids and bestsmen dancing', 'celebration of wedding party', 'everyone is dancing', 'weddings party with colorful lights', 'everyone laughing and dancing in the wedding party'],
'special_moments':['bride and groom in a great moment together', 'bride and groom doing a great photosession together', ' bride and groom with a fantastic standing looking to each other with beautiful scene', 'Couple with a scenic view', 'bride and groom kissing each other in a photoshot', 'bride and groom holding hands', 'bride and groom half hugged for a speical photo moment'],
'family_portraits':['family together with bride and groom standing posing for photo', 'bride with her parents posing  as a group photo', 'groom with his parents standing as posing group'],
'detail_shots':['table setting', 'decor','invitation letter','weddings rings' ,'cake cutting without showing faces', 'bride and groom cut a cake without showing faces','brides shoes or sandals', 'empty tables and chairs inside hall'],
'wedding_dinner':['dinner food on table', 'guests setting around tables having dinner', 'food on table and people eating'],
'groom_and_bride_first_dance':['groom and brides dancing together solo','first dance of groom and bride', 'bride and groom dancing alone' ],
'walking_down_the_aisle': ['bride with her father walking the aisle','groom waiting the bride in the aisle','groom looking at the bride walking the aisle','bridesmaids walking with groomsmen in the aisle','kids with flowers walking the aisle','couples walking the aisle', ],

}

small_intervals_queries = {
    'cluster_0': {'bride_preparation_before_wedding': category_queries['bride_preparation_before_wedding'],'groom_preparation_before_wedding':category_queries['groom_preparation_before_wedding'],'wedding_reception_before_ceremony': category_queries['wedding_reception_before_ceremony'],'couple_arrival_before_ceremony': category_queries['couple_arrival_before_ceremony'],'walking_down_the_aisle' : category_queries['walking_down_the_aisle'], 'photo_session_bride_and_groom': category_queries['photo_session_bride_and_groom']},
    'cluster_1': {'walking_down_the_aisle' : category_queries['walking_down_the_aisle'],'wedding_ceremony': category_queries['wedding_ceremony'], 'photo_session_bride_and_groom': category_queries['photo_session_bride_and_groom']},
    'cluster_2': {'detail_shots':category_queries['detail_shots'],'dancing':category_queries['dancing'],'family_portraits':category_queries['family_portraits'], 'wedding_dinner':category_queries['wedding_dinner'],'groom_and_bride_first_dance':category_queries['groom_and_bride_first_dance'], 'bride_bridesmaids_photosession':category_queries['bride_bridesmaids_photosession'], 'groom_bestmen_photosession':category_queries['groom_bestmen_photosession'] },
}


large_intervals_queries = {
    'cluster_0': {},
    'cluster_1': {},
    'cluster_2': {},
    'cluster_3': {},
    'cluster_4': {},
    'cluster_5': {},

}