import os

def generate_clean_dataset():
    os.makedirs('dataset', exist_ok=True)
    file_path = 'dataset/llms-full.txt'
    
    dataset_content = """[ANDROID_MANIFEST]
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.INTERNET" />
    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="TurnIt"
        android:theme="@style/Theme.TurnIt">
        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>

[GRADLE_KOTLIN_DSL]
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
android {
    namespace = "com.turnit.ai"
    compileSdk = 33
    defaultConfig {
        applicationId = "com.turnit.ai"
        minSdk = 24
        targetSdk = 33
        versionCode = 1
        versionName = "1.0"
    }
    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
}
dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.compose.ui:ui:1.5.0")
    implementation("androidx.compose.material3:material3:1.1.2")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.6.2")
}

[COMPOSE_CHAT_UI]
@Composable
fun ChatScreen(messages: List<ChatMessage>) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp)
    ) {
        items(messages) { message ->
            val alignment = if (message.isUser) Alignment.End else Alignment.Start
            val bgColor = if (message.isUser) Color.Transparent else Color(0x33FFFFFF) // Glassmorphism base
            
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp),
                contentAlignment = alignment
            ) {
                Surface(
                    shape = RoundedCornerShape(16.dp),
                    color = bgColor,
                    modifier = Modifier.glassmorphismEffect()
                ) {
                    Text(
                        text = message.text,
                        modifier = Modifier.padding(12.dp),
                        color = Color.White
                    )
                }
            }
        }
    }
}

[COMPOSE_ANIMATION]
@Composable
fun AnimatedRGBBorder(content: @Composable () -> Unit) {
    val infiniteTransition = rememberInfiniteTransition()
    val angle by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(2000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        )
    )
    
    Box(
        modifier = Modifier
            .drawBehind {
                rotate(angle) {
                    drawRect(
                        brush = Brush.sweepGradient(
                            colors = listOf(Color.Red, Color.Green, Color.Blue, Color.Red)
                        ),
                        style = Stroke(width = 4.dp.toPx())
                    )
                }
            }
            .padding(4.dp)
    ) {
        content()
    }
}

[COMPOSE_NAVIGATION_DRAWER]
@Composable
fun TurnItDrawer(onNavigate: (String) -> Unit) {
    ModalDrawerSheet {
        Spacer(Modifier.height(12.dp))
        NavigationDrawerItem(
            icon = { Icon(Icons.Default.Add, contentDescription = "New Chat") },
            label = { Text("New Chat") },
            selected = false,
            onClick = { onNavigate("new_chat") }
        )
        NavigationDrawerItem(
            icon = { Icon(Icons.Default.History, contentDescription = "History") },
            label = { Text("History") },
            selected = false,
            onClick = { onNavigate("history") }
        )
        NavigationDrawerItem(
            icon = { Icon(Icons.Default.Settings, contentDescription = "API Key Settings") },
            label = { Text("API Key Settings") },
            selected = false,
            onClick = { onNavigate("settings") }
        )
    }
}
"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(dataset_content.strip())
    
    print(f"Successfully generated pure Android/Compose dataset at {file_path}")
    print(f"Total characters: {len(dataset_content)}")

if __name__ == "__main__":
    generate_clean_dataset()
