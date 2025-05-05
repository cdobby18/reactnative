import { View, Text, FlatList, TouchableOpacity, Image } from 'react-native';
import React from 'react';
import { router } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { images } from '../../constants';

const Home = () => {
  return (
    <SafeAreaView className="bg-black flex-1">
      <FlatList
        ListHeaderComponent={() => (
          <View className="my-6 px-4">
            <View className="flex-row items-center justify-between mt-8 mb-14">
              <View className="flex-row items-center space-x-4">
                {images.profile && (
                  <Image
                    source={images.profile}
                    style={{ width: 48, height: 48, borderRadius: 24 }}
                  />
                )}
                <View>
                  <Text className="font-pmedium text-[#FF3A4A] font-psemibold">
                    Welcome!
                  </Text>
                  <Text className="text-2xl font-psemibold text-white">
                    Guest
                  </Text>
                </View>
              </View>
              <TouchableOpacity
                onPress={() => router.push('/')}
                activeOpacity={0.7}
              >
                <Text className="text-2xl text-[#FF3A4A] font-psemibold">
                  Logout
                </Text>
              </TouchableOpacity>
            </View>
            <View className="justify-center items-center px-10 py-16 mt-20">
              <Text className="text-6xl text-[#FF3A4A] font-bold text-center">
                Welcome to POWERLIFT!
              </Text>
              <Text className="text-3xl text-white mt-6 text-center">
                Start your Progress
              </Text>
            </View>
            <View className="mt-8 items-center">
              <Image
                source={images.logoSmall}
                style={{ width: 36, height: 40 }}
              />
            </View>
          </View>
        )}
      />
    </SafeAreaView>
  );
};

export default Home;