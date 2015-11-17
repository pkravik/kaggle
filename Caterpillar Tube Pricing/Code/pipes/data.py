__author__ = 'p_kravik'

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

pd.set_option('display.width', 1000)
import math


class Specs(object):
    def __init__(self, folder):
        self.folder = folder
        self.data = None
        self.reshaped_data = None
        self.features = None

    def read_raw_file(self):
        data = pd.read_csv(self.folder + 'specs.csv')
        return data

    def reshape_raw_data(self):
        data = self.read_raw_file()

        reshaped_data_array = []
        for spec_num in map(str, range(1, 11)):
            cols = ['tube_assembly_id'] + ['spec' + spec_num]
            reshaped_data_array.append(data[cols])

        def rename_stuff(x):
            x.columns = ['tube_assembly_id', 'spec']
            return x

        map(rename_stuff, reshaped_data_array)

        reshaped_data = pd.concat(reshaped_data_array, axis=0).dropna(axis=0).sort('tube_assembly_id')
        self.data = reshaped_data
        return self

    def clean(self):
        data = self.reshape_raw_data()

    def create_features(self):
        data = self.data
        data['weight'] = 1
        features = pd.pivot_table(data, 'weight', 'tube_assembly_id', 'spec', aggfunc='sum')
        features.fillna(0, inplace=True)
        self.features = features

        num_specs = data.groupby('tube_assembly_id').size()
        feat = num_specs.reset_index()
        feat.columns = ['tube_assembly_id', 'num_specs']

        feat = feat.merge(self.features.reset_index(), on='tube_assembly_id')

        return feat

    def pca_features(self, num_components=5):
        data = self.features
        ids = data.index
        pca = PCA(num_components)
        pca.fit(data)
        feat = pd.DataFrame(pca.transform(data))
        feat.columns = ["spec_pca_" + str(x) for x in range(1, num_components + 1)]
        feat['tube_assembly_id'] = ids

        return feat

    def straight_up(self):
        data = self.read_raw_file()
        component_vars = ['spec' + str(x) for x in range(1, 11)]
        for var in component_vars:
            lbl = LabelEncoder()
            data[var] = lbl.fit_transform(data[var])

        return data.fillna(0)


class BillOfMaterials(object):
    def __init__(self, folder):
        self.folder = folder
        self.data = None
        self.reshaped_data = None
        self.features = None

    def read_raw_file(self):
        self.data = pd.read_csv(self.folder + "bill_of_materials.csv")
        return self

    def reshape_raw_data(self):
        data = self.data

        comp_num_range = range(1, 9)

        reshaped_data_array = []

        for comp_num in map(str, comp_num_range):
            cols = ['tube_assembly_id'] + ['quantity_' + comp_num, 'component_id_' + comp_num]
            reshaped_data_array.append(data[cols])

        def rename_stuff(x):
            x.columns = ['tube_assembly_id', 'quantity', 'component_id']
            return x

        map(rename_stuff, reshaped_data_array)

        reshaped_data = pd.concat(reshaped_data_array, axis=0).dropna(axis=0).sort('tube_assembly_id')

        self.reshaped_data = reshaped_data

        return self

    @staticmethod
    def create_boolean_version_var(df):
        for var in ['unique_feature', 'orientation', 'groove', 'plating']:
            if var in df.columns:
                df['boolean_' + var] = df[var] == 'Yes'

        return df

    @staticmethod
    def create_quantity_weighted_var(df, prefix):
        for var in ['unique_feature', 'orientation', 'groove', 'plating']:
            if prefix + var in df.columns:
                df[prefix + 'total_' + var] = df.quantity * df[prefix + 'boolean_' + var]
                df.drop(prefix + 'boolean_' + var, axis=1, inplace=True)

        if prefix + 'weight' in df.columns:
            df[prefix + 'total_weight'] = df.quantity * df[prefix + 'weight']

        return df

    def get_component_data(self):
        components = pd.read_csv(self.folder + 'components.csv')
        df = self.reshaped_data

        df = df.merge(components, on='component_id', how='inner')

        feat = df.groupby('tube_assembly_id').agg({'name': {'num_unique_component_name': lambda x: x.nunique()},
                                                   'component_type_id': {'num_unique_component_type_id': lambda x: x.nunique()}})

        feat.columns = feat.columns.droplevel()
        feat = feat.reset_index()
        feat.fillna(-1, inplace=True)
        return feat

    def clean_adaptors(self):
        adaptors = pd.read_csv(self.folder + 'comp_adaptor.csv')
        self.create_boolean_version_var(adaptors)
        adaptors.drop('component_type_id', axis=1, inplace=True)

        categorical_var = ['end_form_id_1', 'connection_type_id_1', 'end_form_id_2', 'connection_type_id_2',
                           'unique_feature', 'orientation']
        adaptors = self.encode_cat_variables(adaptors, categorical_var)
        adaptors.fillna(0, inplace=True)
        adaptors.columns = ['component_id'] + ['adap_' + x for x in adaptors.columns[1:]]

        # unique at the tube_assembly_id level, don't need to aggregate
        return adaptors

    def adaptor_features(self):
        df = self.reshaped_data
        df = df.merge(self.clean_adaptors(), on='component_id', how='inner')
        self.create_quantity_weighted_var(df, 'adap_')
        df.fillna(-1, inplace=True)
        df = df.drop(['quantity', 'component_id'], axis=1)
        return df

    def clean_boss(self):
        boss = pd.read_csv(self.folder + 'comp_boss.csv')
        self.create_boolean_version_var(boss)
        boss.drop('component_type_id', axis=1, inplace=True)
        categorical_var = ['type', 'connection_type_id', 'outside_shape', 'base_type', 'groove', 'unique_feature',
                           'orientation']
        boss = self.encode_cat_variables(boss, categorical_var)
        boss.fillna(0, inplace=True)
        boss.columns = ['component_id'] + ['boss_' + x for x in boss.columns[1:]]

        return boss

    def boss_features(self):
        # not unique at tube_assembly_id, gotta aggregate
        # most have 1, max is 5 boss
        df = self.reshaped_data
        df = df.merge(self.clean_boss())
        self.create_quantity_weighted_var(df, 'boss_')

        categorical_var = ['type', 'connection_type_id', 'outside_shape', 'base_type']
        continuous_var = ['height_over_tube', 'bolt_pattern_long', 'bolt_pattern_wide', 'base_diameter',
                          'shoulder_diameter', 'weight']
        dummy_vars = ['unique_feature', 'orientation']

        grouped_stats = self.generic_aggregation_stats('boss_', categorical_var, continuous_var, dummy_vars)

        feat = df.groupby('tube_assembly_id').agg(grouped_stats)

        feat.columns = feat.columns.droplevel()
        feat.fillna(-1, inplace=True)
        feat = feat.reset_index()

        return feat

    @staticmethod
    def generic_aggregation_stats(prefix=None, categorical_var=None, continuous_var=None, dummy_vars=None):
        # For continuous variables do min, max, and mean
        # For categorical variables, do num of unique and the most freq
        grouped_stats = {}
        if continuous_var is not None:
            for var in continuous_var:
                grouped_stats[prefix + var] = {'min_' + prefix + var: lambda x: x.min(),
                                               'max_' + prefix + var: lambda x: x.max(),
                                               'mean_' + prefix + var: lambda x: x.mean()}
                if var == 'weight':
                    grouped_stats[prefix + 'total_' + var] = {prefix + 'total_' + var: lambda x: x.sum()}

        if categorical_var is not None:
            for var in categorical_var:
                grouped_stats[prefix + var] = {'num_' + prefix + var: lambda x: x.nunique(),
                                               'most_freq_' + prefix + var: lambda x: x.value_counts().head(1)}

        if dummy_vars is not None:
            for var in dummy_vars:
                grouped_stats[prefix + var] = {'num_' + prefix + var: lambda x: x.sum(),
                                               'pct_' + prefix + var: lambda x: x.sum() / len(x)}

                if var in ['unique_feature', 'orientation', 'groove', 'plating']:
                    grouped_stats[prefix + 'total_' + var] = {prefix + 'total_' + var: lambda x: x.sum()}


        grouped_stats['quantity'] = {prefix + 'total_components': lambda x: x.sum(),
                                     prefix + 'unique_component': lambda x: x.size}


        return grouped_stats

    def clean_elbow(self):
        elbow = pd.read_csv(self.folder + 'comp_elbow.csv')
        self.create_boolean_version_var(elbow)
        elbow.drop('component_type_id', axis=1, inplace=True)
        categorical_var = ['mj_class_code', 'mj_plug_class_code', 'groove', 'unique_feature', 'orientation']
        elbow = self.encode_cat_variables(elbow, categorical_var)
        # elbow.fillna(0, inplace=True)
        elbow.columns = ['component_id'] + ['elbow_' + x for x in elbow.columns[1:]]
        return elbow

    def elbow_features(self):
        # also not unique
        # decent amount have 2 and 3 (max is 3)
        df = self.reshaped_data
        df = df.merge(self.clean_elbow())
        self.create_quantity_weighted_var(df, 'elbow_')

        continuous_var = ['bolt_pattern_long', 'bolt_pattern_wide', 'extension_length', 'overall_length', 'thickness',
                          'drop_length', 'elbow_angle', 'weight']
        categorical_var = ['mj_class_code', 'mj_plug_class_code']
        dummy_var = ['groove', 'unique_feature', 'orientation']

        grouped_stats = self.generic_aggregation_stats('elbow_', categorical_var, continuous_var, dummy_var)

        feat = df.groupby('tube_assembly_id').agg(grouped_stats)
        feat.columns = feat.columns.droplevel()
        feat.fillna(-1, inplace=True)
        feat = feat.reset_index()

        return feat

    def clean_floats(self):
        floats = pd.read_csv(self.folder + 'comp_float.csv')
        self.create_boolean_version_var(floats)
        floats.drop('component_type_id', axis=1, inplace=True)
        categorical_var = ['orientation']
        floats = self.encode_cat_variables(floats, categorical_var)
        floats.columns = ['component_id'] + ['float_' + x for x in floats.columns[1:]]
        return floats

    def float_features(self):
        # floats is unique at tube_assembly, no need to aggregate
        df = self.reshaped_data
        df = df.merge(self.clean_floats(), on='component_id', how='inner')
        self.create_quantity_weighted_var(df, 'float_')
        df.fillna(-1, inplace=True)
        df = df.drop(['quantity', 'component_id'], axis=1)
        return df

    def clean_hfl(self):
        # !!!!!!!!!!!!!!!!!!!!!!!!! ###
        # Should figure out the deal with this 'corresponding_shell'
        # !!!!!!!!!!!!!!!!!!!!!!!!!!

        hfl = pd.read_csv(self.folder + 'comp_hfl.csv')
        self.create_boolean_version_var(hfl)
        hfl.drop('component_type_id', axis=1, inplace=True)
        categorical_var = ['coupling_class', 'material', 'plating', 'orientation', 'corresponding_shell']
        hfl = self.encode_cat_variables(hfl, categorical_var)
        hfl.columns = ['component_id'] + ['hfl_' + x for x in hfl.columns[1:]]

        return hfl

    def hfl_features(self):
        # unique at tube_assembly, no need to aggregate
        df = self.reshaped_data
        df = df.merge(self.clean_hfl(), on='component_id', how='inner')
        self.create_quantity_weighted_var(df, 'hfl_')
        df = df.drop(['quantity', 'component_id'], axis=1)
        df = df.fillna(-1)
        return df

    def clean_nut(self):
        # Need to fix the nut thread size at some point I think
        nut = pd.read_csv(self.folder + 'comp_nut.csv')
        self.create_boolean_version_var(nut)
        nut.drop('component_type_id', axis=1, inplace=True)
        categorical_var = ['blind_hole', 'orientation', 'thread_size']
        nut = self.encode_cat_variables(nut, categorical_var)
        nut.columns = ['component_id'] + ['nut_' + x for x in nut.columns[1:]]
        nut.fillna(-1, inplace=True)
        return nut

    def nut_features(self):
        df = self.reshaped_data
        df = df.merge(self.clean_nut(), on='component_id', how='inner')
        self.create_quantity_weighted_var(df, 'nut_')
        categorical_var = ['thread_size']
        continuous_var = ['hex_nut_size', 'seat_angle', 'length', 'thread_pitch', 'diameter', 'weight']
        dummy_var = ['blind_hole', 'orientation']
        grouped_feat = self.generic_aggregation_stats('nut_', categorical_var, continuous_var, dummy_var)
        feat = df.groupby('tube_assembly_id').agg(grouped_feat)
        feat.columns = feat.columns.droplevel()
        feat = feat.reset_index()
        feat.fillna(-1, inplace=True)
        return feat

    def clean_other(self):
        other = pd.read_csv(self.folder + 'comp_other.csv')
        self.create_boolean_version_var(other)
        # just drop aprt name for now, could do some text stuff at some point
        other.drop('part_name', axis=1, inplace=True)
        other.columns = ['component_id'] + ['other_' + x for x in other.columns[1:]]

        return other

    def other_features(self):
        # not unique, between 1 and 6, decent spread
        # need to aggregate
        df = self.reshaped_data
        df = df.merge(self.clean_other(), on='component_id', how='outer')
        self.create_quantity_weighted_var(df, 'other_')
        continuous_var = ['weight']
        grouped_feat = self.generic_aggregation_stats('other_', continuous_var=continuous_var)
        feat = df.groupby('tube_assembly_id').agg(grouped_feat)
        feat.columns = feat.columns.droplevel()
        feat = feat.reset_index()
        feat.fillna(-1, inplace=True)

        return feat

    def clean_sleeve(self):
        sleeve = pd.read_csv(self.folder + 'comp_sleeve.csv')
        self.create_boolean_version_var(sleeve)
        sleeve.drop('component_type_id', axis=1, inplace=True)
        categorical_var = ['connection_type_id', 'unique_feature', 'plating', 'orientation']
        sleeve = self.encode_cat_variables(sleeve, categorical_var)
        sleeve.columns = ['component_id'] + ['sleeve_' + x for x in sleeve.columns[1:]]

        return sleeve

    def sleeve_features(self):
        # small amount have 2
        # need to aggregate
        df = self.reshaped_data
        df = df.merge(self.clean_sleeve(), on='component_id', how='inner')
        self.create_quantity_weighted_var(df, 'sleeve_')
        categorical_var = ['connection_type_id']
        continuous_var = ['length', 'intended_nut_thread', 'intended_nut_pitch', 'weight']
        dummy_var = ['unique_feature', 'plating', 'orientation']
        grouped_features = self.generic_aggregation_stats('sleeve_', categorical_var=categorical_var,
                                                          continuous_var=continuous_var, dummy_vars=dummy_var)
        feat = df.groupby('tube_assembly_id').agg(grouped_features)
        feat.columns = feat.columns.droplevel()
        feat = feat.reset_index()
        feat = feat.fillna(-1)
        return feat

    def clean_straight(self):
        straight = pd.read_csv(self.folder + 'comp_straight.csv')
        self.create_boolean_version_var(straight)
        straight.drop('component_type_id', axis=1, inplace=True)
        categorical_var = ['mj_class_code', 'groove', 'unique_feature', 'orientation']
        straight = self.encode_cat_variables(straight, categorical_var)
        straight.fillna(0, inplace=True)
        straight.columns = ['component_id'] + ['straight_' + x for x in straight.columns[1:]]

        return straight

    def straight_features(self):
        # decent amount have 2
        # need to aggregate
        df = self.reshaped_data
        df = df.merge(self.clean_straight(), on='component_id', how='inner')
        self.create_quantity_weighted_var(df, 'straight_')
        categorical_var = ['mj_class_code']
        continuous_var = ['bolt_pattern_long', 'bolt_pattern_wide', 'head_diameter', 'overall_length', 'thickness',
                          'weight']
        dummy_var = ['groove', 'unique_feature', 'orientation']
        grouped_stats = self.generic_aggregation_stats('straight_', categorical_var, continuous_var, dummy_var)
        feat = df.groupby('tube_assembly_id').agg(grouped_stats)
        feat.columns = feat.columns.droplevel()
        feat = feat.reset_index()
        feat = feat.fillna(-1)
        return feat

    def clean_tee(self):
        tee = pd.read_csv(self.folder + 'comp_tee.csv')
        self.create_boolean_version_var(tee)
        tee.drop('component_type_id', axis=1, inplace=True)
        categorical_var = ['mj_class_code', 'mj_plug_class_code', 'groove', 'unique_feature', 'orientation']
        tee = self.encode_cat_variables(tee, categorical_var)
        tee.columns = ['component_id'] + ['tee_' + x for x in tee.columns[1:]]
        return tee

    def tee_features(self):
        # unique, no need to aggregate
        df = self.reshaped_data
        df = df.merge(self.clean_tee(), on='component_id', how='inner')
        self.create_quantity_weighted_var(df, 'tee_')
        df = df.drop(['quantity', 'component_id'], axis=1)
        df = df.fillna(-1)
        return df

    def clean_threaded(self):
        threaded = pd.read_csv(self.folder + 'comp_threaded.csv')
        self.create_boolean_version_var(threaded)

        # nominal_size_1 has a string in it
        threaded.nominal_size_1 = threaded.nominal_size_1.replace(r'\s+', 9999, regex=True).astype(float)

        threaded.drop('component_type_id', axis=1, inplace=True)
        categorical_var = ['end_form_id_' + str(x) for x in range(1, 5)] + \
                          ['connection_type_id_' + str(x) for x in range(1, 5)] + ['unique_feature', 'orientation']
        threaded = self.encode_cat_variables(threaded, categorical_var)
        threaded.fillna(0, inplace=True)
        threaded.columns = ['component_id'] + ['threaded_' + x for x in threaded.columns[1:]]

        return threaded

    def threaded_features(self):
        # most are one or two, max is 5
        df = self.reshaped_data
        df = df.merge(self.clean_threaded(), on='component_id', how='inner')
        self.create_quantity_weighted_var(df, 'threaded_')
        categorical_var = ['end_form_id_' + str(x) for x in range(1, 5)] + \
                          ['connection_type_id_' + str(x) for x in range(1, 5)]
        continuous_var = ['adaptor_angle', 'overall_length', 'hex_size', 'weight']

        for i in range(1, 5):
            continuous_var = continuous_var + [x + '_' + str(i) for x in
                                               ['length', 'thread_size', 'thread_pitch', 'nominal_size']]
        dummy_var = ['unique_feature', 'orientation']

        grouped_feat = self.generic_aggregation_stats('threaded_', categorical_var, continuous_var, dummy_var)
        feat = df.groupby('tube_assembly_id').agg(grouped_feat)
        feat.columns = feat.columns.droplevel()
        feat = feat.reset_index()
        feat = feat.fillna(-1)
        return feat

    def encode_cat_variables(self, df, categorical_var):
        for var in categorical_var:
            lbl = LabelEncoder()
            df[var] = lbl.fit_transform(df[var])
        return df

    def get_features(self):
        data = self.reshaped_data
        num_components = data.groupby('tube_assembly_id').quantity.agg(
            {'total_components': 'sum', 'num_unique_components': 'size'})
        feat = num_components.reset_index()
        # pca_feat = self.pca_features(10)
        # straight_up = self.straight_up()
        print "merging component category data..."
        feat = feat.merge(self.get_component_data(), on='tube_assembly_id', how='outer')
        print "merging straight up..."
        feat = feat.merge(self.straight_up(), on='tube_assembly_id', how='outer')
        print "merging adaptors..."
        feat = feat.merge(self.adaptor_features(), on='tube_assembly_id', how='outer')
        print "merging boss..."
        feat = feat.merge(self.boss_features(), on='tube_assembly_id', how='outer')
        print "merging elbow..."
        feat = feat.merge(self.elbow_features(), on='tube_assembly_id', how='outer')
        print "merging float..."
        feat = feat.merge(self.float_features(), on='tube_assembly_id', how='outer')
        print "merging hfl..."
        feat = feat.merge(self.hfl_features(), on='tube_assembly_id', how='outer')
        print "merging nut..."
        feat = feat.merge(self.nut_features(), on='tube_assembly_id', how='outer')
        print "merging other..."
        feat = feat.merge(self.other_features(), on='tube_assembly_id', how='outer')
        print "merging sleeve..."
        feat = feat.merge(self.sleeve_features(), on='tube_assembly_id', how='outer')
        print "merging straight..."
        feat = feat.merge(self.straight_features(), on='tube_assembly_id', how='outer')
        print "merging tee..."
        feat = feat.merge(self.tee_features(), on='tube_assembly_id', how='outer')
        print "merging threaded..."
        feat = feat.merge(self.threaded_features(), on='tube_assembly_id', how='outer')

        print "creating_aggregate_features"
        self.create_aggregate_features(feat)

        feat.fillna(-1, inplace=True)
        print "finished with components..."
        return feat

    def create_aggregate_features(self, df):
        stuff = {'adap_': ['unique_feature', 'orientation', 'weight'],
                 'boss_': ['unique_feature', 'orientation', 'weight'],
                 'elbow_': ['groove', 'unique_feature', 'orientation', 'weight'],
                 'float_': ['orientation', 'weight'],
                 'hfl_': ['plating', 'orientation', 'weight'],
                 'nut_': ['orientation', 'weight'],
                 'other_': ['weight'],
                 'sleeve_': ['unique_feature', 'plating', 'orientation', 'weight'],
                 'straight_': ['groove', 'unique_feature', 'orientation', 'weight'],
                 'tee_': ['groove', 'unique_feature', 'orientation', 'weight'],
                 'threaded_': ['unique_feature', 'orientation', 'weight']}

        pieces = {'unique_feature': [],
                  'orientation': [],
                  'weight': [],
                  'groove': [],
                  'plating': []}

        for pre, arr in stuff.iteritems():
            for var in arr:
                pieces[var].append(pre)

        for var, arr in pieces.iteritems():
            cols = [pre + 'total_' + var for pre in arr]
            df.loc[:, cols] = df.loc[:, cols].replace(-1, 0)
            df['combined_' + var] = df[cols].sum(axis=1)

        return df

    def aggregate_reshaped_data(self):
        data = self.reshaped_data
        features = pd.pivot_table(data, 'quantity', 'tube_assembly_id', 'component_id', aggfunc='sum')
        features.fillna(0, inplace=True)
        self.features = features
        return features.reset_index()

    def pca_features(self, num_components=5):
        self.aggregate_reshaped_data()
        data = self.features
        ids = data.index
        pca = PCA(num_components)
        pca.fit(data)
        feat = pd.DataFrame(pca.transform(data))
        feat.columns = ["bill_pca_" + str(x) for x in range(1, num_components + 1)]
        feat['tube_assembly_id'] = ids

        return feat

    def straight_up(self):
        data = self.data
        component_vars = ['component_id_' + str(x) for x in range(1, 9)]
        for var in component_vars:
            lbl = LabelEncoder()
            data[var] = lbl.fit_transform(data[var])

        return data.fillna(0)


class TubeData(object):
    def __init__(self, folder):
        self.folder = folder
        self.data = None
        self.key = ['tube_assembly_id']
        self.categorical_var = ['material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x']
        self.continuous_var = ['diameter', 'wall', 'length', 'num_bends', 'bend_radius', 'num_boss', 'num_bracket',
                               'other']
        self.columns_omitted = None
        self.categorical_encoder = None

    def read_raw_file(self):
        self.data = pd.read_csv(self.folder + "tube.csv")

    def clean(self):
        self.read_raw_file()
        self.get_end_form()
        tube = self.data
        tube = self.fix_data_error(tube)

        key = self.key

        categorical_var = self.categorical_var
        continuous_var = self.continuous_var
        included_var = categorical_var + continuous_var + key
        columns_omitted = [x for x in tube.columns if x not in included_var]
        self.columns_omitted = columns_omitted

        encoder_dict = {}
        for var in categorical_var:
            encoder = LabelEncoder()
            tube[var] = encoder.fit_transform(tube[var])
            encoder_dict[var] = encoder

        self.categorical_encoder = encoder_dict

        self.data = tube[key + categorical_var + continuous_var]
        self.create_new_features()

        return self.data

    def get_end_form(self):
        end_form = pd.read_csv(self.folder + "tube_end_form.csv")
        data = self.data

        for end_type in ['end_a', 'end_x']:
            end_form.columns = [end_type, 'forming_' + end_type]
            self.categorical_var += ['forming_' + end_type]
            data = data.merge(end_form, on=end_type, how='left')
            # data['forming_' + end_type].fillna('No')

        self.data = data

    def create_new_features(self):
        # Cross section area
        data = self.data
        radius = data.diameter / 2
        data.loc[:, 'cross_section_area'] = math.pi * (-1 * data.wall ** 2 + 2 * radius * data.wall)
        data.loc[:, 'total_volume'] = data.cross_section_area * data.length
        data.loc[:, 'diameter_over_thickness'] = data.diameter / data.wall
        data.loc[:, 'bend_over_wall'] = data.bend_radius / data.wall
        data.loc[:, 'random_bend_ratio'] = data.diameter_over_thickness / data.bend_over_wall
        data.loc[:, 'wall_thinning'] = data.diameter / (data.diameter + data.bend_radius)
        data.loc[:, 'length_per_bend'] = data.length / data.num_bends
        data.loc[:, 'length_per_bend_radius'] = data.length_per_bend / data.bend_radius
        data = data.replace([np.inf, -np.inf], np.nan)

        self.data = data

    def fix_data_error(self, df):
        # List below is lengths that need to be corrected
        # Corrected "length"

        corrections = {'TA-00152': 19,
                       'TA-00154': 75,
                       'TA-00156': 24,
                       'TA-01098': 10,
                       'TA-01631': 48,
                       'TA-03520': 46,
                       'TA-04114': 135,
                       'TA-17390': 40,
                       'TA-18227': 74,
                       'TA-18229': 5}

        for tube, length in corrections.iteritems():
            df.loc[df.tube_assembly_id == tube, 'length'] = length

        # Diameter is ODD
        return df


class CombinedData(object):
    def __init__(self, folder):
        self.folder = folder
        self.spec_object = None
        self.bill_object = None
        self.tube_object = None
        self.combined_data = None

    def create_data(self):
        folder = self.folder
        # Specifications
        spec = Specs(folder)
        spec.clean()
        spec_features = spec.straight_up()
        self.spec_object = spec

        # Bill of materials (list of components)
        bill = BillOfMaterials(folder)
        bill.read_raw_file()
        bill.reshape_raw_data()
        bill_features = bill.get_features()
        self.bill_object = bill

        # Tube data
        tube = TubeData(folder)
        tube_data = tube.clean()
        self.tube_object = tube

        tube_features = pd.merge(tube_data, spec_features, 'outer', 'tube_assembly_id')
        tube_features = pd.merge(tube_features, bill_features, 'outer', 'tube_assembly_id')
        self.create_new_features(tube_features)
        tube_features.fillna(0, inplace=True)
        self.combined_data = tube_features

    def merge_features(self, train):
        tube_features = self.combined_data
        train_data = pd.merge(train, tube_features, on='tube_assembly_id', how='left')
        train_data.fillna(0, inplace=True)
        return train_data

    def create_new_features(self, df):
        # df['weight_per_volume'] = df.combined_weight / df.total_volume
        # df['weight_per_length'] = df.combined_weight / df.length

        df['orientation_per_length'] = df.combined_orientation / df.length
        df['plating_per_length'] = df.combined_plating / df.length
        df['component_per_length'] = df.total_components / df.length
        df['unique_features_per_length'] = df.combined_unique_feature / df.length
        df['sleeve_length_per_tube_length'] = df.mean_sleeve_length / df.length

        df = df.replace([np.inf, -np.inf], np.nan)

        return df


class BasicData(object):
    def __init__(self, folder):
        self.folder = folder

    def clean_data(self, train):
        train['num_tiers'] = train.groupby('tube_assembly_id').quantity.transform('size')
        train['num_quantity_tiers'] = train.groupby(['tube_assem    bly_id', 'quote_date']).quantity.transform(lambda x: x.nunique())

        categorical_vars = ['supplier', 'bracket_pricing']
        for categorical_var in categorical_vars:
            train[categorical_var] = LabelEncoder().fit_transform(train[categorical_var])

        train, created_var = self.gen_new_variables(train)

        date_var = 'quote_date'
        train[date_var] = pd.to_datetime(train[date_var])
        # train[date_var + '_dayofweek'] = train[date_var].dt.dayofweek
        train[date_var + '_month'] = train[date_var].dt.month
        train[date_var + '_week'] = train[date_var].dt.week
        train[date_var + '_year'] = train[date_var].dt.year

        key = ['tube_assembly_id']
        continuous_var = ['annual_usage', 'min_order_quantity', 'quantity', 'num_tiers', 'num_quantity_tiers', 'quote_date_month',
                          'quote_date_week', 'quote_date_year'] #, 'quote_date_dayofweek']
        categorical_var = ['bracket_pricing', 'supplier']

        train['supplier'] = train.groupby('supplier').supplier.transform('size')

        return train[key + continuous_var + categorical_var + created_var + ['cost']]

    def gen_new_variables(self, train):
        created_var = ['annual_usage_per_quantity', 'annual_usage_per_min_quantity',
                       'min_order_quantity_per_quantity']  # , 'tier_place']

        train['annual_usage_per_quantity'] = train.annual_usage / train.quantity
        train['annual_usage_per_min_quantity'] = train.annual_usage / (train.min_order_quantity + 0.00001)
        train['min_order_quantity_per_quantity'] = train.min_order_quantity / train.quantity

        train['num_dates'] = train.groupby('tube_assembly_id').quote_date.transform(lambda x: x.nunique()).astype(int)
        train['num_annual_usage'] = train.groupby('tube_assembly_id').annual_usage.transform(lambda x: x.nunique())

        tier_types = train.groupby(['quote_date', 'tube_assembly_id']).quantity.agg({'tier_type': lambda x: ' '.join([str(z) for z in x])})
        rain = train.merge(tier_types.reset_index())

        place_in_tier = train.groupby(['quote_date', 'tube_assembly_id']).quantity.transform(lambda x: range(1, x.size + 1))
        train['tier_place'] = place_in_tier

        train.pct_max_tier = train.groupby(['quote_date', 'tube_assembly_id']).quantity.transform(lambda x: x/x.max())

        # Should have date place I think

        # enc = LabelEncoder()

        return train, created_var

    def get_data(self):
        folder = self.folder

        train = pd.read_csv(folder + 'train_set.csv')
        test = pd.read_csv(folder + 'test_set.csv')
        test = test.drop('id', 1)

        outcome = train.cost
        combined = self.clean_data(pd.concat([train, test]))

        clean_train = combined[combined.cost.notnull()]
        clean_test = combined[combined.cost.isnull()]

        clean_train = clean_train.drop('cost', 1)
        clean_test = clean_test.drop('cost', 1)

        return clean_train, clean_test, outcome

    def get_weird_regression_data(self):
        folder = self.folder
        train = pd.read_csv(folder + 'train_set.csv')
        test = pd.read_csv(folder + 'test_set.csv')
        test = test.drop('id', 1)
        train['num_quantity_tiers'] = train.groupby(['tube_assembly_id', 'quote_date']).quantity.transform(lambda x: x.nunique())
        test['num_quantity_tiers'] = test.groupby(['tube_assembly_id', 'quote_date']).quantity.transform(lambda x: x.nunique())

        def fit_regression_and_get_r2(df):
            clf = LinearRegression()
            unique_quantity = df.quantity.nunique()
            num_obs = df.quantity.size
            target = df.quantity * df.cost
            clf.fit(np.reshape(df.quantity.values, (num_obs, 1)), np.reshape(target.values, (num_obs, 1)))
            score = r2_score(target, clf.predict(np.reshape(df.quantity, (num_obs,1))))
            slope = clf.coef_[0][0]
            constant = clf.intercept_[0]
            return pd.Series({'r2': score, 'slope': slope, 'constant': constant, 'supplier': df.supplier.iloc[0], 'num_tiers':num_obs, 'num_quantity': unique_quantity})

        train_data = train[train.num_quantity_tiers == 8].groupby(['tube_assembly_id', 'quote_date']).apply(fit_regression_and_get_r2)
        train_data = train_data.reset_index()
        outcomes = train_data[['constant', 'slope']]
        train_data = train_data[['tube_assembly_id', 'supplier']]

        lbl = LabelEncoder()
        train_data['supplier'] = lbl.fit_transform(train_data.supplier)

        other_train = train.groupby('tube_assembly_id').agg({'annual_usage': 'mean', 'min_order_quantity': 'mean'})

        train_data = train_data.merge(other_train.reset_index())

        other_features = CombinedData(folder)
        other_features.create_data()

        train_data = other_features.merge_features(train_data)

        return train_data, outcome


def get_train_data():
    folder = "../Data/Raw/"

    basic = BasicData(folder)
    train_data, test_data, outcome = basic.get_data()

    other_features = CombinedData(folder)
    other_features.create_data()

    train_data = other_features.merge_features(train_data)
    test_data = other_features.merge_features(test_data)

    print "Complete!"

    return train_data, test_data, outcome


def get_weird_train_data():
    folder = "../Data/Raw/"

    basic = BasicData(folder)
    train_data, outcomes = basic.get_weird_regression_data()

    return train_data, outcomes

class KLabelFolds():
    def __init__(self, labels, n_folds=3):
        self.labels = labels
        self.n_folds = n_folds

    def __iter__(self):
        unique_labels = self.labels.unique()
        cv = cross_validation.KFold(len(unique_labels), self.n_folds)
        for train, test in cv:
            test_labels = unique_labels[test]
            test_mask = self.labels.isin(test_labels)
            train_mask = np.logical_not(test_mask)
            yield (np.where(train_mask)[0], np.where(test_mask)[0])


from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV

if __name__ == '__main__':

    train_data, outcome = get_weird_train_data()
    train_id = train_data.pop('tube_assembly_id')
    outcome1 = outcome['slope']
    outcome2 = outcome['constant']

    a_train, a_test, b_train, b_test, id_train, id_test = train_test_split(train_data, outcome, train_id, test_size = 0.25, random_state=42)

    gbm1 = GradientBoostingRegressor(n_estimators=100, max_depth=8, min_samples_leaf=2, random_state=421)
    gbm1.fit(a_train.values, b_train.slope)
    print "train error: %f" % mean_squared_error(outcome1, gbm.predict(train_data.values))

    # cross_val_score(gbm, train_data.values, outcome1, scoring='mean_squared_error', cv=5)

    gbm2 = GradientBoostingRegressor(n_estimators=100, max_depth=8, min_samples_leaf=2, random_state=421)
    gbm2.fit(a_train.values, b_train.constant)
    print "train error: %f" % mean_squared_error(outcome2, gbm.predict(train_data.values))
    # cross_val_score(gbm, train_data.values, outcome2, scoring='mean_squared_error', cv=5)

    slopes = gbm1.predict(a_test)
    constant = gbm2.predict(a_test)
    print "slope test error: %f" % mean_squared_error(b_test.slope, slopes)
    print "constant test error: %f" % mean_squared_error(b_test.constant, constant)

    stuff = pd.DataFrame({'tube_assembly_id': id_test,
                         'slope': slopes,
                         'constant': constant})

    test = pd.read_csv(folder + 'train_set.csv')
    test = test.merge(stuff, on='tube_assembly_id')

    test['prediction'] = (test.constant + test.quantity * test.slope)/test.quantity
    print "RMSLE for weight regression thing %f" %rmsle(test.cost, test.prediction)

    def rmsle(true, pred):
        return np.sqrt(np.mean((np.log1p(true) - np.log1p(pred))**2))

    train_data, test_data, outcome = get_train_data()
    train_id = train_data.pop('tube_assembly_id')
    test_id = test_data.pop('tube_assembly_id')

    new_outcome = outcome * train_data.quantity

    np.random.seed(42)
    blah = train_id.groupby(train_id).agg(lambda x: np.random.uniform())
    blah.sort()
    num_tubes = blah.size
    n_folds = 5
    labels = pd.Series([x * n_folds / (num_tubes + 1) for x in range(1, num_tubes + 1)])

    cv_labels = pd.DataFrame({'tube_assembly_id': blah.index,
                              'label': labels})

    groups = pd.DataFrame(train_id).merge(cv_labels, on='tube_assembly_id').label
    cv_iterator = LeaveOneLabelOut(groups)

    log_outcome = np.log(outcome + 1)

    # Models

    rf = RandomForestRegressor(n_estimators=1000, max_depth=8, max_features=0.75, min_samples_leaf=2, random_state=421)
    # rf = RandomForestRegressor(n_estimators=1, max_depth=10, min_samples_leaf=5)

    rf.fit(train_data.values, log_outcome)
    print "train error: %f" % np.sqrt(mean_squared_error(log_outcome, rf.predict(train_data.values)))
    print "CV error: %f" % (np.mean(np.sqrt(
        -1 * cross_val_score(rf, train_data.values, log_outcome, scoring='mean_squared_error', cv=cv_iterator))))

    gbm = GradientBoostingRegressor(n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=421)

    gbm.fit(X_train, y_train)
    gbm_pred = gbm.predict(X_test)
    mean_squared_error(y_test, gbm_pred)

    gbm.fit(train_data.values, new_outcome)

    gbm.fit(train_data.values, log_outcome)
    print "train error: %f" % np.sqrt(mean_squared_error(outcome, gbm.predict(train_data.values) / train_data.quantity))
    print "CV error: %f" % (np.mean(np.sqrt(
        -1 * cross_val_score(gbm, train_data.values, log_outcome, scoring='mean_squared_error', cv=cv_iterator))))

    var_importance = pd.DataFrame({'var': train_data.columns, 'sig': gbm.feature_importances_}).sort('sig',
                                                                                                     ascending=False)
    var_importance
    var_importance.to_csv(folder + 'importance2.csv')

    def score_func(y, y_pred, **kwargs):
        return np.sqrt(mean_squared_error(y, y_pred))

    from sklearn.metrics import make_scorer

    scorer = make_scorer(score_func, greater_is_better=False)
    cross_val_score(gbm, train_data.values, outcome, scorer, cv_iterator, 1)

    params = [{'max_depth': [5, 8, 10],
               'n_estimators': [100],
               'learning_rate': [0.1, 1, 0.5],
               'min_samples_leaf': [2, 5],
               'max_features': [None]}]

    clf = GridSearchCV(gbm, params, scoring='mean_squared_error', cv=cv_iterator, n_jobs=3, verbose=1, iid=False)

    clf.fit(train_data.values, log_outcome)

    clf.grid_scores_

    rf.fit(train_data, log_outcome)
    rf.predict(train_data)
    predictions = rf.predict(test_data)

    def rmsle(truth, prediction):
        diff = np.square(np.log(prediction + 1) - np.log(truth + 1))

    for train_index, test_index in cv_iterator:
        print(len(train_index), len(test_index))
        x_train, x_test = train_data.iloc[train_index], train_data.iloc[test_index]
        y_train, y_test = outcome.iloc[train_index], outcome.iloc[test_index]

        # rf.fit(x_train, y_train)
        # print(np.sqrt(mean_squared_error(y_test, rf.predict(x_test))))

    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression
    def fit_regression_and_get_r2(df):
        clf = LinearRegression()
        unique_quantity = df.quantity.nunique()
        num_obs = df.quantity.size
        target = df.quantity * df.cost
        clf.fit(np.reshape(df.quantity.values, (num_obs, 1)), np.reshape(target.values, (num_obs, 1)))
        score = r2_score(target, clf.predict(np.reshape(df.quantity, (num_obs,1))))
        slope = clf.coef_[0][0]
        constant = clf.intercept_[0]
        return pd.Series({'r2': score, 'slope': slope, 'constant': constant, 'supplier': df.supplier.iloc[0], 'num_tiers':num_obs, 'num_quantity': unique_quantity})

results = train[train.num_quantity_tiers == 8].groupby(['tube_assembly_id', 'quote_date']).apply(fit_regression_and_get_r2)
results.r2.hist()
results[results.r2 < 0.99].num_quantity.hist()
results[results.r2 > 0.99].num_quantity.hist()
results[results.r2 > 0.99]

train[train.tube_assembly_id == 'TA-16268']

plt.figure(1)
results[results.slope < 40].slope.hist()
plt.figure(2)
test_results[test_results.slope < 40].slope.hist()

plt.figure(1)
results[(results.constant < 25) & (results.constant > 15)].constant.hist()
plt.figure(2)
test_results[(test_results.constant < 25) & (test_results.constant > 15)].constant.hist()

plt.figure(1)
results[(results.constant < 40) & (results.slope < 40)].plot(kind='scatter', x='constant', y='slope', ylim = [0, 30], xlim=[0, 30])
plt.figure(2)
test_results[(test_results.constant < 40) & (test_results.slope < 40)].plot(kind='scatter', x='constant', y='slope', ylim = [0, 30], xlim=[0, 30])

# Clearly there is something going on






