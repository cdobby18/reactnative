import { StatusBar } from 'expo-status-bar';
import { ScrollView, Text, View, Image } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import CustomButton from '../components/CustomButton';

export default function Index() {
  return (
    <SafeAreaView className="bg-black h-full">
      <ScrollView contentContainerStyle={{ height: '100%' }}>
        <View className="w-full justify-center items-center min-h-[85vh] px-4">
         
          <Text className="text-3xl text-white font-bold text-center">
            Welcome to
          </Text>
          <Text className="text-6xl text-[#FF3A4A] font-bold mb-5">POWERLIFT</Text>

          <CustomButton
            title="Login as a Guest"
            handlePress={() => router.push('/home')}
            containerStyles="w-full mt-5"
          />

          <CustomButton
            title="Continue with Email"
            handlePress={() => router.push('/sign-in')}
            containerStyles="w-full mt-5"
          />
        </View>
      </ScrollView>
      <StatusBar backgroundColor="#161622" style="light" />
    </SafeAreaView>
  );
}
