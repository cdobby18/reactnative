import { View, Image } from 'react-native';
import { Tabs } from 'expo-router';
import { icons } from '../../constants';

const TabIcon = ({ icon, color, name, focused }) => {
  return (
    <View style={{ alignItems: 'center', justifyContent: 'center', gap: 4 }}>
      <Image
        source={icon}
        resizeMode="contain"
        tintColor={color}
        style={{ width: 32, height: 32, marginTop: 4 }} 
      />
    </View>
  );
};

const TabsLayout = () => {
  return (
    <View style={{ flex: 1, backgroundColor: 'black' }}> 
      <Tabs
        screenOptions={{
          tabBarShowLabel: false,
          tabBarActiveTintColor: '#D0321D',
          tabBarStyle: {
            height: 70,
            backgroundColor: 'transparent', 
          },
        }}
      >
        <Tabs.Screen
          name="home"
          options={{
            title: 'Home',
            headerShown: false,  
            tabBarIcon: ({ color, focused }) => (
              <TabIcon
                icon={icons.home}
                color={color}
                name="Home"
                focused={focused}
              />
            ),
          }}
        />
        <Tabs.Screen
          name="profile"
          options={{
            title: 'Profile',
            headerShown: false,  
            tabBarIcon: ({ color, focused }) => (
              <TabIcon
                icon={icons.user}
                color={color}
                name="User"
                focused={focused}
              />
            ),
          }}
        />
        <Tabs.Screen
          name="analysis"
          options={{
            title: 'Analysis',
            headerShown: false,  
            tabBarIcon: ({ color, focused }) => (
              <TabIcon
                icon={icons.lift}
                color={color}
                name="Exercise"
                focused={focused}
              />
            ),
          }}
        />
        <Tabs.Screen
          name="feedback"
          options={{
            title: 'Feedback',
            headerShown: false,  
            tabBarIcon: ({ color, focused }) => (
              <TabIcon
                icon={icons.radar}
                color={color}
                name="Feedback"
                focused={focused}
              />
            ),
          }}
        />
      </Tabs>
    </View>
  );
};

export default TabsLayout;