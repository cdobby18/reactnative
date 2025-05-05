import React from 'react';
import { View, Text, FlatList, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';

export default function Feedback() {
  const navigation = useNavigation();

  return (
    <SafeAreaView className="bg-black h-full">
      <FlatList
        ListHeaderComponent={
          <View className="my-6 px-4 space-y-4">
            <View className="justify-start items-start mt-6">
              <Text className="text-3xl font-semibold text-gray-100 mb-1">
                Progress Tracking
              </Text>
              <Text className="">
                Guest
              </Text>
            </View>

            <Text className="text-xl font-bold text-[#FF3A4A] text-center">
              PROGRESS TRACKING
            </Text>

            <View className="bg-white/20 p-10 rounded-lg items-center w-full self-center">
              <TouchableOpacity className="bg-[#FF3A4A] py-3 px-6 rounded-lg">
                <Text className="text-white font-bold text-lg">DEADLIFT</Text>
              </TouchableOpacity>
            </View>

            <View className="bg-white/20 p-10 rounded-lg items-center w-full self-center">
              <TouchableOpacity className="bg-[#FF3A4A] py-3 px-6 rounded-lg">
                <Text className="text-white font-bold text-lg">SQUAT</Text>
              </TouchableOpacity>
            </View>

            <View className="bg-white/20 p-10 rounded-lg items-center w-full self-center">
              <TouchableOpacity className="bg-[#FF3A4A] py-3 px-6 rounded-lg">
                <Text className="text-white font-bold text-lg">BENCH PRESS</Text>
              </TouchableOpacity>
            </View>
          </View>
        }
        data={[]} 
        renderItem={null}
      />
    </SafeAreaView>
  );
}
