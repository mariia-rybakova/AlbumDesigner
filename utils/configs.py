CONFIGS = {'DEBUG': True,

           # Debug info saving flags
           'save_files': {
               'groups': False,
               'spreads': False,
               'top_k': 5
               },

           'queries_file': r'files/queries_embeddings_v1.pkl',
           'queries_file_v2': r'files/queries_embeddings_v2.pkl',
           'queries_file_v3': r'files/queries_embeddings_v3.pkl',
           'designs_json_file_path': r'files/designs.json',
           'image_loading_timeout': 30,
           'min_split_score':2,
           'max_img_split':2,
           'max_total_spreads':20,
           'max_group_spread':3,
           'max_imges_per_spread':24,
           # Photos-per-spread at/above which a class counts as "densely packed"
           # (e.g. 'dancing' at 24). Such groups are diluted by the min-floor
           # expansion only as a last resort, so dense classes keep their density
           # unless nothing else can reach the floor (e.g. a dancing-only gallery).
           'expansion_dense_threshold': 12,
           'min_per_spread': 4,
           'max_per_spread': 38,
           'distance_weight': 10,
           'size_weight': 0.9,
           'split_group_penalty': 0.01,
           'max_size_for_normalize': 44,
           'min_size_for_normalize': 2,
           'split_homogenous_group_penalty': 0.1,
           'crop_penalty': 0.01,
           'color_mix': 0.0001,
           'double_page_color_mix': 0.00000001,
           'class_mix': 0.00001,
           'orientation_mix': 0.1,
           'spread_score_threshold': 0.01,
           'partition_score_threshold': 100,
           'MaxCombs': 1000,
           'MaxCombsLargeGroups': 100,
           'MaxOrientedCombs': 300,
           'top_imges_for_cover' : 3,
           'max_reading_workers':2,
           'cropping_workers':2,
           'max_lay_workers': 1,
           "max_photos_group":12,
           'wedding_merge_images_number':2,
           'collection_name': 'aigeneratealbumdto',
           'visibility_timeout':1200,
           'products_json_location':'pictures/photostore/32/settings/products.json.txt',
           'design_pack_base':'pictures/photostore/32/ext/designs',
           'architect_base':'pictures/photostore/32/ext/productgroups',

           'merge_limit_times': 3,
           'none_limit_times': 5,

            #Qdrant
            "QDRANT_HOST": "10.0.44.13",
            "QDRANT_COLLECTION": {1:"ImageEmbedding_V1_new",
                                  2:"ImageEmbedding_V2_new",},
            #mongo DB
            "DB_NAME": "aimongo",
            "STATUS_COLLECTION_NAME": "aiprojectstatusdal",
            "DB_CONNECTION_STRING_VAR": "MongoConnectionString",


           #selection
           'content_threads': 4,
           'small_groups': 3,
           'max_number_images': 130,
           'person_score': 0.0000001,
           'similarity_score': 0.00001,
           'class_matching_zero_score': 0.0001,
           'class_matching_penalty': 0.001,
           'small_groups_not_to_select': 3,
           'grays_scale_limit': 2,
           'person_count_percentage': 0.20,
           'small_gallery_number': 15,
           'events_disallowing_small_images': ['settings', 'vehicle', 'rings', 'food', 'accessories', 'entertainment',
                                               'dancing',
                                               'wedding dress', 'kiss'],
           'total_target_images': 130,
           'layouts': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24],
           'FACE_FAR_THRESHOLD': 0.4,
           'FACE_M_THRESHOLD': 0.7,
           'BODY_FAR_THRESHOLD': 0.7,
           'BODY_MEDIUM_THRESHOLD': 0.4,
           'ε': 1e-9,
           'focus_csv_path':r'files/focus_csv.csv',
           'bin_name_dictionary':{
                'weddingDress':'wedding_dress',
                'brideGettingReady':'bridegettingready',
                'groomGettingReady':'groomgettingready',
               'tableSetting':'table_setting',
               'holdingHands':'holdinghands',
               'softLight': 'softlight',
           },
        'weights' :{
            'class': 0.2,
            'similarity': 0.2,
            'person': 0.4,
            'tags': 0.1,
            'rank': 0.2,
            'user_rating': 0.3
         },
        'user_rating_max_scale': 5,
        'density_factors' : {1: 0.5, 2: 0.75, 3: 1, 4: 1.5, 5: 2.0},
        # The ceremony exit: guests showering the couple as they leave
        # (confetti, petals, bubbles, rice, sparklers). Detected as a temporal
        # burst after the ceremony climax that also looks like a send-off; the
        # visual evidence is mandatory, sequence alone never tags.
        # Thresholds calibrated on galleries 49994361 / 49995684 (send-offs
        # present, burst means 0.42 and 0.44) against 47981912 / 53496523
        # (none, gallery maxima 0.30 and 0.42 but no qualifying burst).
        # -- enrich.ceremony_anchor: the kiss -------------------------------
        # The kiss is one of the climax signals the anchor is built from, so it
        # is found around the anchor rather than after it. It has real
        # vocabulary in the query bank, so no concept bin is needed.
        'kiss_radius': 60,
        'kiss_max_gap': 3,
        'kiss_max_photos': 6,
        'kiss_eligible_labels': ('ceremony', 'kiss', 'bride and groom', 'other', 'None'),
        # The query bank's two kiss subqueries do not fire on every gallery: on
        # 47981912 the only kiss frame is labelled "officiant leading wedding
        # ceremony" and carries no identity, so there is no label evidence at
        # all. A concept bank finds it -- that frame scores at the 99.5th
        # percentile of its gallery. Frames clearing the floor join the
        # subquery-matched ones as candidates; ranking is concept-led, since
        # proximity to the anchor cannot separate a kiss from the vows (the
        # anchor is the median of both).
        'kiss_concept': 'ceremony_kiss',
        'kiss_concept_floor': {2: 0.40, 1: 1.0},
        # Ablated: on v2 the subquery route is redundant -- the concept alone
        # finds the same frames on all four galleries, including the one whose
        # only kiss frame carries no kiss subquery. It is kept because on v1
        # kiss_concept_floor is inert, making the subquery the ONLY route in;
        # removing it would silently kill kiss detection on every v1 gallery.
        'kiss_subquery_bonus': 0.05,

        # -- enrich.ceremony_anchor: the processional -----------------------
        # Entering the ceremony, so the anchor is read as an UPPER bound -- and
        # the bound is the ceremony START, not the climax: "before the ceremony"
        # means before it begins. Bounding at the anchor instead pulled in
        # mid-ceremony vows and officiant frames as "processional".
        'aisle_lead_in': 80,
        'aisle_upper_overlap': 20,
        'aisle_max_gap': 4,
        # After the winning run is chosen, absorb solo frames this close to it.
        # The groom's "waiting at the altar" frame sits a few frames off the end
        # of his walk in and would otherwise be a run of one.
        'aisle_extend_gap': 10,
        'aisle_min_photos': 2,
        'aisle_max_photos': 6,
        'aisle_eligible_labels': ('walking the aisle', 'ceremony', 'bride', 'groom',
                                  'bride and groom', 'bride party', 'groom party',
                                  'other', 'portrait'),
        'aisle_concepts': {'bride': 'bride_aisle', 'groom': 'groom_aisle'},
        # Identity is the mandatory signal. Candidate runs are then RANKED, not
        # gated -- there is no concept floor. Weighted sum of:
        #   subquery   fraction of the run carrying a processional subquery
        #   proximity  1/(1+distance/half) from the ceremony start
        #   concept    the run's mean CLIP concept score, used RAW
        #
        # Raw, deliberately. Normalising the concept term across runs turns a
        # meaningless 0.055 spread in the v1 space into a full point, which
        # picked the groom's prep shots over his walk in. Used raw it
        # contributes in proportion to the signal it carries -- ~0.2 of
        # separation in v2, almost none in v1 -- so it self-calibrates across
        # embedding spaces. Ranking within one gallery is scale-free; only
        # absolute thresholds break across spaces, which is why the floor is
        # gone. Verified against ground truth on 47981912 (a v1 gallery):
        # bride 4/4, groom 5/6.
        # Ablated across the four validation galleries: dropping the subquery
        # term changes exactly one of eight aisle picks -- 49995684's bride goes
        # from the run carrying "bride walking down aisle with father" x4 to a
        # nearer unlabelled run. Marginal, but it is the deciding vote in the
        # one case where proximity misleads, and it costs nothing.
        'aisle_rank_weights': {'subquery': 1.0, 'proximity': 1.0, 'concept': 1.0},
        'aisle_proximity_half': 10,
        # Distance to the ceremony start is ASYMMETRIC: a run beginning after
        # the ceremony has started is charged this multiple per position it
        # overruns. It has to be a penalty rather than a hard cut-off -- the
        # detected ceremony start is fuzzy, and on one gallery the real bride
        # walk begins 10 positions after it while on another a mid-ceremony run
        # 9 positions after it is not the processional at all. 6 separates them;
        # 3 was not enough, because the wrong run's concept score is higher.
        'aisle_after_penalty': 6.0,
        # A floor on the WINNING run, keyed by embedding space. Ranking is
        # scale-free, but a floor is not -- so it applies only where it works.
        #   v2: 0.22. Ground truth spans a wide range -- 0.44-0.49 on three
        #       galleries but 0.244-0.308 on 47981912, whose real processional
        #       a 0.32 floor rejected. The prep runs a gallery falls back to
        #       when it has none score 0.05-0.18, so the gate still does its
        #       job with room to spare.
        #   v1: 0 (ungated). No floor survives this space: on the one v1 gallery
        #       with ground truth, the groom's real walk in scores BELOW his own
        #       gallery's median, so any floor -- absolute or relative -- rejects
        #       the correct answer. v1 relies on the ranking alone and accepts
        #       the fallback described in the tests.
        'aisle_score_floor': {2: 0.22, 1: 0.0},

        # -- enrich.ceremony_anchor: the send-off ---------------------------
        'send_off_concept': 'send_off',
        'send_off_eligible_labels': ('ceremony', 'walking the aisle', 'other', 'bride and groom'),
        'send_off_photo_floor': {2: 0.35, 1: 1.0},
        'send_off_burst_floor': {2: 0.38, 1: 1.0},
        'send_off_min_photos': 5,
        'send_off_max_gap': 3,
        'send_off_back_slack': 40,
        'send_off_horizon': 250,

        # How many of the ceremony's 'yes' classes -- the kiss, the two
        # processionals, the send-off -- have to turn up before they are worth
        # a page of the album between them. One special moment is a photo; two
        # or more is a page. Below this they are charged to the ceremony's own
        # allowance instead. See src/pipeline/select/allocation.py.
        'ceremony_yes_min_classes': 2,

        'MAX_PERSON_COMBINATION': 10000,
        'use_rebalance_spreads': False,

}

merge_content_priority = {
    'accessories': 0.9,
    'bride': 0.0001,
    'bride and groom': 0.0001,
    'bride getting dressed': 0.5,
    'bride party': 0.3,
    'cake cutting': 0.9,
    'ceremony': 0.0001,
    'couple': 0.0001,
    'dancing': 0.9,
    'detail': 0.91,
    'entertainment': 0.92,
    'first dance': 0.001,
    'food': 0.92,
    'full party': 0.001,
    'getting hair-makeup': 0.92,
    'groom': 0.0001,
    'groom party': 0.3,
    'inside vehicle': 0.92,
    'invite': 0.92,
    'kiss': 0.2,
    'pet': 0.92,
    'portrait': 0.0001,
    'rings': 0.92,
    'settings': 0.92,
    'speech': 0.92,
    'suit': 0.92,
    'vehicle': 0.92,
    'very large group': 0.0001,
    'walking the aisle': 0.0001,
    'wedding dress': 0.92,
    'other': 0.92,
    'nan': 0.92,
}
# Content priority scores
split_content_priority = {
    'accessories': 0.0001,
    'bride': 0.9,
    'bride and groom': 0.9,
    'bride getting dressed': 0.5,
    'bride party': 0.3,
    'cake cutting': 0.001,
    'ceremony': 0.8,
    'couple': 0.7,
    'dancing': 0,
    'detail': 0.001,
    'entertainment': 0.001,
    'first dance': 0.6,
    'food': 0.001,
    'full party': 0.001,
    'getting hair-makeup': 0.001,
    'groom': 0.9,
    'groom party': 0.3,
    'inside vehicle': 0.001,
    'invite': 0.001,
    'kiss': 0.2,
    'pet': 0.001,
    'portrait': 0.02,
    'rings': 0.0001,
    'settings': 0.0001,
    'speech': 0.02,
    'suit': 0.0001,
    'vehicle': 0.0001,
    'very large group': 0.0001,
    'walking the aisle': 0.2,
    'wedding dress': 0.0001,
    'other': 0.000001,
    'nan': 0.00001,
}

relations = {'brideAndGroom':{
    'bride and groom': (10, 0.2),

    'bride': (7, 0.1),
    'groom': (10, 0.001),

    'bride party': (8, 0.5),
    'groom party': (8, 0.5),
    'full party': (4, 0.5),

    'large_portrait': (4, 0.5),
    'small_portrait': (7, 0.5),
    'portrait': (5, 0.1),
    'very large group': (8, 0.5),
    'walking the aisle': (6, 0.4),

    'bride getting dressed': (6, 0.5),
    'first dance': (2, 0.2),
    'cake cutting': (2, 0.1),
    'ceremony': (10, 0.2),
    'couple': (0, 0),
    'dancing': (24, 0.5),

    'entertainment': (0, 0.2),

    'kiss': (2, 0.6),
    'pet': (0, 0.5),

    'accessories': (1, 0.01),
    'settings': (5, 0.1),
    'speech': (4, 0.5),

    'detail': (5, 0.5),
    'getting hair-makeup': (4, 0.1),
    'food': (5, 0.5),
    'other': (0, 0),
    'None':(0,0),
    'invite': (1, 0.5),
    'wedding dress':(2,0.01),
    'vehicle': (2,0.01),
    'inside vehicle':(2,0.01),
    'suit':(1, 0.5),
    'rings':(1,0.5),
    'may kiss bride': (1, 0.2),
    'send off': (4, 0.3),
    'bride walking the aisle': (2, 0.2),
    'groom walking the aisle': (2, 0.2),
    'bride and groom with parents': (1, 0.2),
    'groom with his parents': (1, 0.2),
    'bride with her parents': (1, 0.2),
    'parents portrait': (1, 0.2)


},
    'parents': {
        'bride and groom': (8, 0.2),

        'bride': (4, 0.1),
        'groom': (4, 0.001),

        'bride party': (4, 0.5),
        'groom party': (4, 0.5),
        'full party': (4, 0.5),
        'portrait': (5, 0.1),
        'very large group': (10, 0.5),
        'walking the aisle': (4, 0.4),

        'bride getting dressed': (3, 0.5),
        'first dance': (2, 0.2),
        'cake cutting': (2, 0.1),
        'ceremony': (15, 0.2),
        'couple': (0, 0),
        'dancing': (12, 0.5),

        'entertainment': (2, 0.2),

        'kiss': (2, 0.6),
        'pet': (1, 0.5),

        'accessories': (1, 0.01),
        'settings': (8, 0.1),
        'speech': (8, 0.5),

        'detail': (8, 0.5),
        'getting hair-makeup': (2, 0.1),
        'food': (10, 0.5),
        'other': (0, 0.5),
        'invite': (1, 0.5),
        'wedding dress': (2, 0.01),
        'vehicle': (2, 0.01),
        'inside vehicle': (2, 0.01),
        'suit': (0, 0.5),
        'rings': (1, 0.5),
        'may kiss bride': (1, 0.2),
        'send off': (4, 0.3),
        'bride walking the aisle': (2, 0.2),
        'groom walking the aisle': (2, 0.2),
        'bride and groom with parents': (1, 0.2),
        'groom with his parents': (1, 0.2),
        'bride with her parents': (1, 0.2),
        'parents portrait':(1,0.2)

    }
,'everyoneElse':{
    'bride and groom': (15, 0.2),

    'bride': (4, 0.1),
    'groom': (4, 0.001),
    'bride party': (8, 0.5),
    'groom party': (8, 0.5),
    'full party': (6, 0.5),
    'portrait': (5, 0.1),
    'very large group': (6, 0.5),
    'walking the aisle': (10, 0.4),

    'bride getting dressed': (6, 0.5),
    'first dance': (4, 0.2),
    'cake cutting': (4, 0.1),
    'ceremony': (10, 0.2),
    'couple': (0, 0),
    'dancing': (24, 0.5),

    'entertainment': (3, 0.2),

    'kiss': (2, 0.6),
    'pet': (0, 0.5),

    'accessories': (1, 0.01),
    'settings': (5, 0.1),
    'speech': (5, 0.5),

    'detail': (5, 0.5),
    'getting hair-makeup': (4, 0.1),
    'food': (5, 0.5),
    'other': (0, 0.5),
    'invite': (1, 0.5),
    'wedding dress':(1,0.01),
    'vehicle': (1,0.01),
    'inside vehicle':(1,0.01),
    'suit':(1, 0.5),
    'rings': (1, 0.5),
    'may kiss bride': (1, 0.2),
    'send off': (4, 0.3),
    'bride walking the aisle': (2, 0.2),
    'groom walking the aisle': (2, 0.2),
    'bride and groom with parents': (1, 0.2),
    'groom with his parents': (1, 0.2),
    'bride with her parents': (1, 0.2),
    'parents portrait': (1, 0.2)

}}


limit_imgs = {
    'bride and groom': 9,

    'bride': 7,
    'groom': 7,

    'bride party': 8,
    'groom party': 8,
    'full party': 4,
    'portrait': 10,
    'very large group': 6,
    'walking the aisle': 7,
    'bride getting dressed': 4,
    'first dance': 5,
    'cake cutting': 5,
    'ceremony': 8,
    'couple': 0,
    'dancing': 24,
    'entertainment': 4,
    'kiss': 6,
    'pet': 2,
    'accessories': 2,
    'settings': 8,
    'speech': 8,
    'detail': 10,
    'getting hair-makeup': 4,
    'food': 6,
    'other':0,
    'invite': 1,
    'wedding dress': 1,
    'rings': 1,
    'vehicle': 1,
    'parents portrait':2,
    'may kiss bride':1,
    'send off':4,
    'bride walking the aisle':2,
    'groom walking the aisle':2

}

label_list = [
    'accessories',
    'bride',
    'bride and groom',
    'bride getting dressed',
    'bride party',
    'cake cutting',
    'ceremony',
    'couple',
    'dancing',
    'detail',
    'entertainment',
    'first dance',
    'food',
    'full party',
    'getting hair-makeup',
    'groom',
    'groom party',
    'inside vehicle',
    'invite',
    'kiss',
    'pet',
    'portrait',
    'rings',
    'settings',
    'speech',
    'suit',
    'vehicle',
    'very large group',
    'walking the aisle',
    'wedding dress',
    'other',
    'two brides',
    'two grooms',
]

# image selection
spreads_required_per_category = {
    'bride and groom': 2,
    'bride': 2,
    'groom': 2,
    'bride party': 1,
    'groom party': 1,
    'full party': 1,
    'large_portrait': 2,
    'small_portrait': 1,
    'portrait': 1,
    'very large group': 2,
    'walking the aisle': 1,
    'bride getting dressed': 0.5,
    'first dance': 1,
    'cake cutting': 1,
    'ceremony': 1,
    'couple': 0,
    'dancing': 1,
    'entertainment': 1,
    'kiss': 1,
    'pet': 0,
    'accessories':1,
    'settings': 1,
    'speech': 1,
    'detail': 1,
    'getting hair-makeup': 0.5,
    'food': 1,
    'other': 0,
    'invite': 0,
    'None':0,
    'wedding dress': 1,
    'vehicle':0,
    'inside vehicle':0,
    'may kiss bride':1,
    'send off':1,
    'bride walking the aisle':1,
    'groom walking the aisle':1,
    'parents portrait':1,

}

priority_categories = ['bride and groom',
                       'bride','groom','ceremony', 'may kiss bride','bride party',
    'groom party','full party','dancing','large_portrait', 'parents portrait', 'send off',
    'bride walking the aisle','groom walking the aisle',
    'portrait','very large group',
    'walking the aisle',
    'first dance','cake cutting','bride getting dressed',
    'couple',
    'entertainment',
    'kiss',
    'pet',
    'accessories',
    'settings',
    'speech',
    'detail',
    'getting hair-makeup',
    'food',
    'invite',
    'wedding dress',
    'vehicle',
    'inside vehicle']


min_images_per_category = {
    'bride and groom': 4,
    'bride': 4,
    'groom': 4,
    'bride party': 3,
    'groom party': 3,
    'full party': 1,
    'large_portrait': 2,
    'small_portrait': 1,
    'portrait': 10,
    'very large group': 2,
    'walking the aisle': 4,
    'bride getting dressed': 2,
    'first dance': 1,
    'cake cutting': 2,
    'ceremony': 8,
    'couple': 0,
    'dancing': 10,
    'entertainment': 1,
    'kiss': 1,
    'pet': 0,
    'accessories': 1,
    'settings': 4,
    'speech': 4,
    'detail': 4,
    'getting hair-makeup': 2,
    'food': 4,
    'other': 0,
    'invite': 1,
    'None': 0,
    'wedding dress': 1,
    'vehicle': 1,
    'inside vehicle': 1,
    'rings':1,
    'parents portrait':2,
    'may kiss bride':1,
    'send off':4,
    'bride walking the aisle':2,
    'groom walking the aisle':2

}


selection_threshold = {
    'bride and groom': 0.14,
    'bride': 0.3,
    'groom': 0.005,
    'bride party': 0.14,
    'groom party': 0.14,
    'full party': 0.14,
    'portrait': 0.2,
    'very large group': 0.2,
    'walking the aisle': 0.3,
    'bride getting dressed': 0.2,
    'first dance': 0.2,
    'cake cutting': 0.2,
    'ceremony': 0.14,
    'couple': 0.2,
    'dancing': 0.14,
    'entertainment': 0.14,
    'kiss': 0.2,
    'pet': 0.14,
    'accessories':0.5,
    'settings': 0.3,
    'speech': 0.3,
    'detail': 0.3,
    'getting hair-makeup': 0.2,
    'food': 0.2,
    'other': 0.15,
    'invite': 0.15,
    'None':0.15,
    'wedding dress': 0.2,
    'vehicle':0.15,
    'inside vehicle':0.15,
    'rings':0.15,
    'suit':0.15,
    'may kiss bride':0.15,
    'send off':0.15,
    'bride walking the aisle':0.15,
    'groom walking the aisle':0.15,
    'parents portrait':0.15,
}


# Reserved separator for special-group name suffixes (e.g. 'None|0|14'). Must
# never appear in a real class name, so content-key resolution can strip the
# suffix by splitting on this symbol without corrupting names that legitimately
# contain '_' (e.g. 'large_portrait', 'small_portrait').
SPECIAL_GROUP_SEP = '|'

# Content classes that mean "the classifier had nothing to say" rather than a
# real subject. Groups with one of these are the "special" groups: they are the
# ones `album_tools.split_groups` singles out to be tagged 'class|idx', get
# their own merge limit ('none_limit_times'), all count as one context in
# `merge.count_contexts`, and are the only classes whose groups do not treat a
# shared label as shared content when merging. Every one of those rules reads
# this tuple, so they cannot drift apart.
SPECIAL_CONTENT_CLASSES = ('None', 'other')

