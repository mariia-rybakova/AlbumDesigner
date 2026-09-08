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

        # The getting-ready categories pick their subject by bride_id, the
        # identity enrich.identities already resolved, falling back to a
        # 'bride' subquery match only when she is attached to no frame at all.
        # Before this the substring match was the *primary* rule, which admits
        # bridesmaid / bridal suite / bride's mother -- and on a gallery where
        # seven of ten 'getting hair-makeup' frames carry no subquery text, the
        # photo that reached the album contained neither the bride nor anyone
        # related to her.
        #
        # Set False to pick the way the pre-refactor monolith did; the
        # equivalence tests use that to stay meaningful.
        'bride_prep_by_identity': True,

        # -- enrich.parents ------------------------------------------------
        #
        # Resolve the parents as *identities*, with an explicit inconclusive
        # outcome, instead of classifying photos by head count + gender + an
        # age offset. Set 'by_identity' False for the old rule -- kept because
        # the equivalence tests need a way back to the pre-refactor behaviour,
        # not because it is worth running (its age window accepts +3..+27 years
        # and so selects the sibling band, and its social-circle test collapses
        # to "either candidate appears in any circle at all").
        #
        # Every threshold here is set to fail toward silence. A parent we do
        # not name costs a category that would have been budgeted anyway; a
        # stranger we do name goes on the family spread.
        'parents': {
            'by_identity': True,

            # A parents portrait is of the parents *with the couple* and
            # nobody else. Requiring only "a partner and a parent" admits the
            # big posed group shots -- the bride and her mother among twenty
            # guests -- which the content model files under `portrait` and
            # which are not what the family spread is for.
            #
            # `max_extra_people` counts identified faces beyond the couple and
            # the named parents. `max_unidentified_faces` is the slack against
            # the frame's own face and body counts, because `persons_ids` lists
            # only who was recognised: a crowd in which three people were
            # identified still reads as a crowd, and the identity set alone
            # cannot see it. One, so a single unrecognised face -- a turned
            # head, a child -- does not disqualify a genuine family portrait.
            'max_extra_people': 0,
            'max_unidentified_faces': 1,

            # Noise floor. Below this an identity has no measurable pattern.
            'min_appearances': 5,
            # At most this many identities per side -- two parents, or three to
            # allow a step-parent, never a whole extended family.
            'max_per_side': 2,

            # Side assignment. `min_side_share` of a candidate's
            # one-partner-only frames must fall on one side; below that we
            # cannot say whose parent they are, which is itself a false mark.
            'min_side_frames': 4,
            'min_side_share': 0.7,

            # A parent sits in the older part of the gallery's own identities.
            # This is a rank, never an offset in years: face-age estimators
            # regress toward the mean, so the gap compresses while the
            # ordering survives. Hard requirement -- nothing overrides it.
            # 0.70 rather than 0.65 because that is where the real
            # separation sits: on 53459898 a 50-year-old at rank 0.66 with no
            # evidence beyond being photographed with the bride was clearing a
            # score floor tuned to exclude her, while the groom's mother at
            # rank 0.84 was not. Encode the requirement here, not in the score.
            'min_age_rank': 0.70,

            # The score a candidate must clear, and the separation the last
            # accepted candidate must have over the first rejected one. If two
            # candidates are within `min_margin` we cannot tell which is the
            # parent, so the side goes unresolved rather than guessing.
            'min_score': 0.50,
            'min_margin': 0.10,

            # Saturation points: this many occurrences score a full 1.0.
            'prep_full': 4,
            'aisle_full': 3,
            # The party penalty is relative to the most party-heavy candidate
            # in the same gallery, because absolute counts do not travel: the
            # groom's father on 53459898 has 10 `groom party` frames against
            # the groomsmen's 45-53, and in a suit he is not visually separable
            # from them. Below this many frames the gallery has no party
            # coverage to compare against and the penalty is dropped.
            'min_party_reference': 8,
            # Frames alone with one partner. Family reach double figures; a
            # guest stays near zero.
            'own_side_full': 10,

            # Query augmentation of age. Scored only on photos where the
            # candidate is one of at most `max_ids_for_attribution` identities,
            # so the image-level cosine is about them; then shrunk by
            # sample/(sample+query_prior), because the raw delta ranks a
            # candidate with three attributable photos above one with thirty.
            'max_ids_for_attribution': 3,
            'query_prior': 10,
            'query_full': 0.12,

            # The officiant is old, at the ceremony, on neither side, and
            # confined to a narrow band of the day.
            'officiant_span': 0.15,
            'officiant_ceremony': 10,

            # Circles this size or smaller read as a household rather than a
            # guest list; a circle shared with another *old* candidate is how
            # the two parents of one side corroborate each other.
            'max_circle_size': 4,

            'weights': {
                'age_rank': 0.40,
                'own_side': 0.20,
                'prep': 0.20,
                'aisle': 0.15,
                'duo_dance': 0.25,
                'circle': 0.15,
                'query': 0.15,
                # Negative terms. The party count is the only indicator nearly
                # exclusive to the confusion class, so it is allowed to sink a
                # candidate on its own.
                'party': 0.60,
                'officiant': 0.50,
            },
        },

        # The album's opening and closing photos. Scored rather than picked off
        # a priority ladder, because the ladder had no way to say "this frame
        # matches the subquery I asked for and is still a bad photo".
        #
        # On 53507032 it opened *and* closed the album on one frame: a
        # confetti-line shot with six faces, the groom looking away and the
        # bride's face cut off at the edge. Two separate faults produced it.
        # Landscape was a hard pre-filter applied before the couple test, so a
        # 48-frame candidate base collapsed to the single landscape frame in
        # it; and with only that frame left, the opening ladder found no match,
        # fell through to the rank fallback, and the fallback handed back the
        # same photo the closing ladder had already taken.
        'covers': {
            # Both faces, not just both identities. `persons_ids` is built from
            # face clusters, so a name in it means a face was recognised -- but
            # a photo carrying both names can still be one clear face and one
            # profile at the frame edge. Two detected faces is the cheap
            # version of "both of them are actually in the picture". Relaxed
            # rather than enforced when it would empty the candidate base.
            'min_faces': 2,

            # Quality, from concepts already published for both model versions
            # -- no new bin, so this needs no blob write. `portrait` and
            # `softlight` speak to the photograph, the rest to the moment.
            # Their mean ranked the frame the album actually opened on 32nd of
            # its own 48 candidates, which is the judgement the ladder had no
            # way to make.
            'quality_concepts': ('portrait', 'smiling', 'affection',
                                 'romance', 'softlight', 'happiness'),

            'weights': {
                # The dominant term, and deliberately so: subquery affinity
                # says what the frame is *of*, quality says whether it is worth
                # putting on the cover, and the second is what was missing.
                'quality': 1.00,
                'subquery': 0.60,
                # `image_order` is a rank where 0 is best; the term is inverted
                # where it is applied.
                'rank': 0.40,
                # A cover box is a single large box, so a frame that fills it
                # is preferred -- a preference, never a filter. This is what
                # the hard landscape pre-filter became.
                'orientation': 0.25,
                # Faces that read at cover size. Beyond the couple, more faces
                # means a crowd, so this is a penalty on excess.
                'crowd': 0.30,
            },

            # Faces beyond the couple before `crowd` starts to bite.
            'crowd_slack': 1,
            # The orientation a single-box cover fills best.
            'preferred_orientation': 'landscape',

            # The two covers must be different photos, and far enough apart in
            # the day to read as a beginning and an end. A share of the
            # candidate positions, not minutes: the gallery may not be one
            # continuous session, and `_pick_cover_subset` counts for the same
            # reason. Distinctness holds even at zero.
            'min_separation': 0.25,
        },

        # How many albums one gallery produces. A temporary lever: Phase 3 of
        # docs/multi_album_plan.md replaces it with `enrich.variants`, which
        # derives the count from the gallery rather than from config, because
        # how many albums are worth making depends on what enrich found. 1 is
        # the shipped behaviour and the only value exercised in production.
        'albums': {
            'count': 1,
            # Reseed both global RNGs identically before each album. Off in
            # production: the pipeline samples randomly in several places
            # (combination sampling in spreads_layout, cover choice in
            # album_tools, medoids in time_orientation_selection, the
            # non-wedding selection), so seeding would move a single album's
            # output. On, it is what makes N identical variants comparable.
            'seed': None,
        },

        # `select.narrator`: the albumNarrator policy composing a non-wedding
        # album -- selection and page grouping in one pass. Off by default,
        # because it needs a checkpoint on disk and it changes what a
        # non-wedding album is; turning it on is a deliberate act.
        'narrator': {
            # On for the population it was built for: a non-wedding gallery
            # where the service is being asked to choose. It still declines far
            # more often than it answers -- weddings keep the budgeted CP-SAT
            # path, a v1 gallery cannot be read at 768-d, and any decline falls
            # through to `smart_non_wedding_selection` exactly as before.
            'enabled': True,

            # The trained policy. 46 MB, and loaded once per process. Read with
            # `weights_only=False` because the checkpoint carries the training
            # config the network is rebuilt from -- which makes it executable
            # content, so it must live somewhere only we can write.
            'checkpoint': r'files/narrator/policy.pt',

            # CLIP text-concept axes for the candid / indoor / lighting /
            # bgcolor features (tools/build_attribute_axes.py in the narrator
            # project). Absent, those four features stay at a neutral 0.5 --
            # which is what the narrator itself does without the file, so the
            # album degrades rather than the request failing.
            'attribute_axes': r'files/narrator/attribute_axes.npz',

            # Stochastic rollouts to draw, best-scoring album wins. The
            # narrator measured greedy ~= best-of-8 from v27 on, so 1 is the
            # cheap mode and the default: 0.18 s for a 150-photo gallery at
            # full 768-d dims on a laptop CPU.
            'sample_k': 1,
            'seed': 0,

            # One thread, not every core. The service already processes
            # galleries concurrently, so letting torch fan out per request
            # oversubscribes the box and makes all of them slower.
            'torch_threads': 1,

            # Below this the album is the gallery; composing adds nothing.
            'min_photos': 15,
        },

        # Spread the profile's percentages over the categories the gallery
        # actually has, rather than over the whole profile. The weights sum to
        # 107% and a quarter to a third of that is routinely spent on
        # categories a wedding has none of -- 23%, 25%, 26%, 27%, 31%, 33% on
        # the validation galleries, and 51% on a same-sex one before its solo
        # class was split. Counted in, the present categories asked for only
        # ~70% of the album and the rest came back as shortfall, which the fill
        # loop then absorbed using whatever sat high in focus_csv.csv --
        # including 'other' and 'None' at 0%.
        #
        # Set False to allocate the way the pre-refactor monolith did; the
        # equivalence tests use that to stay meaningful.
        'budget_normalise_present_only': True,

        # Keeping the same shot out of the album twice, when a photographer
        # has uploaded their whole set a second time in black and white or with
        # a colour tone. The test never looks at the colour flag, so a toned
        # copy is caught as readily as a grey one.
        #
        # It is a judgement about the gallery, not about a pair of photos:
        # nothing in the photo table tells a re-export from the next frame of a
        # burst (CLIP cosine and composition both overlap completely), but a
        # duplicated gallery is unmistakable in bulk. Photos sitting in
        # duplicate (capture second, aspect ratio) groups: ~100% on the
        # duplicated gallery 53273032, and 3.8%, 3.1%, 2.1%, 0% on the four
        # ordinary ones. See src/pipeline/enrich/dedupe.py.
        'near_duplicates': {
            'enabled': True,
            # Below this share the groups are read as same-second bursts and
            # nothing is dropped. Two orders of magnitude of daylight either
            # side, so this is not a delicate number.
            'min_gallery_share': 0.5,
            # A group bigger than this cannot be a re-upload set; it is a
            # gallery whose EXIF collapsed onto one value.
            'max_copies_per_shot': 4,
        },

        # select.preselect: which constraints are honoured before the ranked
        # picking runs, and how much of the album each one may claim. Each is
        # switchable on its own so a constraint that costs more than it is worth
        # is one line to disable.
        'preselect': {
            'user_picks': True,
            'identities': True,
            'key_pages': True,
            'yes_categories': True,
            # Guarantee a named identity appears, rather than saturate the album
            # with them: one photo is coverage, and coverage is what selecting a
            # person asks for.
            'photos_per_identity': 1,
        },

        # -- select.pick: the CP-SAT alternative ----------------------------
        # One global constrained solve instead of the per-category loop, after
        # "Algorithms for Constrained Sequence Selection". See
        # src/pipeline/select/cpsat.py for what each term means and which of
        # them the paper does not cover.
        #
        # The weights are the whole argument: `cohesion_weight` pulls picks into
        # runs that read as one moment, the window weights push them across the
        # day, and `shortage_weight` has to dominate both or the solver buys
        # coverage by leaving a class empty. Untuned starting points -- they are
        # scaled against a rank term of at most 1000 per photo.
        'pick_cpsat': {
            # The default picker. `WeddingPicker` remains the fallback and
            # takes over on any failure -- ortools absent, no solution inside
            # the time limit, a modelling mistake -- so this cannot cost an
            # album, and `--loop` on process_gallery forces the old path for a
            # comparison. See docs/cpsat_picker.md.
            'enabled': True,
            'time_limit_seconds': 30,
            'workers': 8,
            'rank_weight': 1,
            'cohesion_weight': 60,
            'cohesion_max_gap': 10,
            'window_weight': 150,
            'class_window_weight': 300,
            'windows': 6,
            'shortage_weight': 4000,
            # Below this many slots a class is spaced rather than windowed:
            # proportional targets round to nothing useful for 2 or 3 photos.
            'sparse_quota': 3,
            'gap_fraction': 0.5,
            # Colour is preferred the way the loop's two pools prefer it, but
            # as a penalty -- one model has one pool.
            'grayscale_penalty': 200,
            # Cohesion rewards neighbours, and the second copy of a shot is a
            # neighbour; above this cosine two frames of a class are exclusive.
            'duplicate_similarity': 0.97,

            # Who a class is about, as a score. `bride` means the bride on
            # her own, `bride and groom` means the two of them and nobody
            # else, `getting hair-makeup` means the bride. The loop says this
            # with hard filters; the model said it nowhere, and it showed --
            # a hair-and-makeup spread of someone else, `groom` frames with no
            # groom in them.
            #
            # Above SCORE_SCALE (1000) on purpose: a matching photo then beats
            # any non-matching one on rank, so the rule decides every class
            # that has matches, while a class with none falls back to rank on
            # its own. That is the loop's `_recover_over_filtering` without the
            # special case -- a hard filter is what makes the recovery
            # necessary in the first place.
            'identity_preference': {
                'enabled': True,
                'weight': 1200,
                # A wrong identity is not a missing one. An empty
                # `persons_ids` is a detection that did not happen and stays
                # neutral; a frame naming someone who is not who the class is
                # about is positively wrong and is charged. Scoring both alike
                # made them equally admissible filler, and on 53459898 a
                # `groom` frame carrying identity 9 reached the album.
                'contradiction_penalty': 1200,
                # And excluded outright in the classes that are definitionally
                # about one person. The penalty alone was outvoted: a
                # contradicted photo still collects the time-coverage rewards
                # for its class and window, +300 and +150, which beat a 1200
                # charge once the rank is in. It only looked sufficient on the
                # validation galleries because those classes had
                # unknown-identity frames to fall back on.
                'exclude_contradictions': True,
                'per_class': {},
            },

            # Drop the temporally isolated photos, as the loop does. A
            # frame with no neighbour within twenty minutes is an outlier
            # rather than part of a moment worth a spread, and nothing else in
            # the model can say so -- an isolated photo is its own bucket in
            # every coverage dimension, so coverage *rewards* taking it.
            #
            # Missing this cost `bride`, `first dance` and `speech` on
            # 52282159: four isolated photos the loop rejects reached the
            # album and restraint scored 0 of 4. It hid for so long because on
            # every other validation gallery temporal narrowing binds at most
            # one class.
            'temporal_narrowing': {
                'enabled': True,
            },

            # `_take_all_distinct`, as a constraint. When a class's supply
            # is at or below its allowance the loop takes everything minus
            # frames of the same people doing the same thing, and no weighting
            # reproduced that: in a three-photo pool every photo is its own
            # bucket in every coverage dimension, so coverage rewards taking
            # all three. Only an exclusion can say two of them are one shot.
            #
            # The supply <= demand condition is what makes it safe. Applied to
            # every class it would cap `dancing` at one photo -- every frame
            # there holds the same couple and carries the same subquery.
            'distinct_shots': {
                'enabled': True,
            },

            # A charge for each extra photo of the same person inside a
            # class -- the counterpart to the `people` coverage dimension, and
            # a different shape. Coverage stops rewarding a second photo of
            # someone; this keeps charging for it, which is what actually
            # pushes a class to spread across faces.
            #
            # Per class, never global: the bride is in most of the gallery, and
            # charging for that would price the album's subject out of her own
            # album. Zero by default for the same reason at class level --
            # every photo in `bride` holds the bride and every photo in `bride
            # and groom` holds both, by construction. It is worth something
            # only where a repeated face is a wasted slot.
            'people_repeat': {
                'enabled': True,
                'weight': 0,
                'free_repeats': 1,
                'max_people': 40,
                # Measured and it does not pay, like the coverage dimensions
                # before it: 8 of 10 right at zero and at a tenth of these
                # values, 7 at a quarter and above, where it starts costing
                # headroom. So it ships inert, with the intended values kept
                # here for whoever revisits it.
                #     'portrait': 250, 'very large group': 250, 'speech': 250,
                #     'full party': 150, 'dancing': 150
                'per_class': {},
            },

            # -- the quota: Phase 4 of docs/cpsat_scoring_plan.md -----------
            #
            # The allowance is a ceiling, not a target. Measured over 60
            # classes, the loop comes in 11 photos short of its budget and
            # never once over, because its diversity passes return fewer items
            # than they were asked for; `sum + shortage == need` cannot express
            # that, since the shortage weight sits above every other term.
            #
            # `admission_cost` is the other half, and neither works alone. In
            # objective units against a rank of score*1000, so 400 means a
            # photo needs a score above 0.4 to earn a page on quality alone --
            # below that it is taken only if the coverage it brings makes up
            # the difference. That is what turns the coverage dimensions from a
            # reshuffle of a fixed count into a reason to stop, and it is why
            # people and content ship at zero weight until this is on.
            #
            # A floor is kept only where one is real: a `yes` class is promised
            # a photo when the thing happened, so coming back empty is a
            # failure rather than restraint.
            'quota_ceiling': {
                'enabled': True,
                # A quantile of the class's own score distribution, because
                # `get_scores` min-max normalises within a class: every class
                # has a photo at 1.0 and one at 0.0, so a flat cost means
                # something different in each. Sweeping a flat one over
                # 53459898 left the count at 143 for every value from 0 to 300
                # and then dropped it to 120 at 400 -- a cliff, not a gradient.
                # 0.2 is where the two halves of the scoreboard balance. At
                # 0 the model fills all 6 photos of similarity headroom but
                # overruns 4 of the 5 classes the loop was right to stop on;
                # at 0.2 it takes 5 of the 6 and correctly leaves 3 of the 5.
                # Chosen on two galleries, so it is an input to the Phase 5
                # fit rather than a tuned value.
                'admission_quantile': 0.2,
                # A floor under the quantile, in raw objective units. Zero
                # unless a gallery needs an absolute bar as well.
                'admission_cost': 0,
            },

            # -- coverage: Phase 1 of docs/cpsat_scoring_plan.md ------------
            #
            # Reward reaching a part of the day instead of penalising drift
            # from a proportional target. Collected once per window, so the
            # second pick in a window earns nothing and the day gets covered
            # without the model being pushed to keep spreading after it is --
            # which is what made it out-spread the loop. Replaces the
            # `window_weight` / `class_window_weight` deviation penalties and
            # `sparse_quota` / `gap_fraction` spacing below; set 'enabled'
            # False to run those instead and measure the difference.
            #
            # `per_class` is the beginning of the w[class][dimension] table the
            # plan is built around. Phase 1 fills in only what the loop already
            # says plainly: the two categories where the user's own pick decides
            # and nothing should be spread, and the single-moment events. Phase
            # 5 fits the rest rather than guessing it.
            'coverage': {
                'enabled': True,
                'time': {
                    'windows': 6,
                    # Against a rank term capped at 1000 per photo.
                    'weight': 300,
                    'global_weight': 150,
                    'per_class': {
                        'accessories': 0,
                        'wedding dress': 0,
                        'cake cutting': 0,
                        'first dance': 0,
                        'kiss': 0,
                        'may kiss bride': 0,
                        'send off': 0,
                        'invite': 0,
                        'rings': 0,
                    },
                },

                # People and content are implemented and measurable, and ship
                # at zero, because measurement says they should. Scaled from
                # 0 to 1x their intended weights, neither improved agreement
                # with the loop over time alone: 81 -> 78-80 of 137 on
                # 53459898, 117 -> 112-119 of 135 on 53147741.
                #
                # The reason is structural rather than a bad number, and it is
                # why Phase 4 has to come first. Coverage rewards *breadth*,
                # but it cannot make the model stop: while every extra photo
                # earns a flat positive rank, more photos is always better, so
                # the solve fills whatever the quota allows. The loop stops
                # because its diversity passes return fewer items than asked.
                # Reproducing that needs the quota as a ceiling *and* a
                # marginal-value rank -- an admission cost per photo, or rank
                # measured net of what is already picked. Until then these two
                # dimensions can only reshuffle a fixed count, and reshuffling
                # away from rank order is what cost the agreement.
                #
                # Intended weights, for the Phase 5 fit to start from rather
                # than invent: people 120 per class and 250 global -- global
                # higher because `person_max_union_selection` exists to get
                # distinct guests into the album *somewhere*, and covering a
                # guest twice buys nothing -- with 300 for the three classes
                # the loop hands to `PersonCoverageStrategy` (`portrait`,
                # `very large group`, `speech`). Content 200 per class and 0
                # global: the same kind of shot in two different classes is not
                # a repetition.
                'people': {
                    'weight': 0,
                    'global_weight': 0,
                    # 89 identities across 27 classes on 53459898; a bucket
                    # holding one photo rewards what the rank term already says.
                    'max_buckets': 40,
                    'per_class': {},
                },
                'content': {
                    'column': 'image_subquery_content',
                    'weight': 0,
                    'global_weight': 0,
                    'max_buckets': 60,
                    'per_class': {},
                },

                # How a photo looks -- what `select_remove_similar` spreads
                # across and what `_add_exclusions` forbids pairwise. Per class
                # only: two frames in different classes that happen to resemble
                # each other are not a repetition.
                #
                # 0.9 rather than the 0.97 the pairwise exclusion uses. That
                # threshold is deliberately tight because it is a *hard*
                # constraint and a false positive deletes a photo outright;
                # coverage is soft, so it can afford to call a wider group "the
                # same shot" and let the ranks argue about it.
                # Ships at zero, like people and content, and for the same
                # measured reason -- see the grid in the plan's §7. At 250 it
                # buys the sixth photo of headroom and gives back a restraint,
                # scoring the same 8 of 11; zero is the safer of the two
                # corners, because breaking restraint puts a photo the loop
                # rejected into the album while missing headroom only leaves a
                # budgeted page unfilled.
                'visual': {
                    'threshold': 0.9,
                    'weight': 0,
                    'global_weight': 0,
                    'max_buckets': 80,
                    'per_class': {
                        'accessories': 0,
                        'wedding dress': 0,
                    },
                },
            },
        },

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

