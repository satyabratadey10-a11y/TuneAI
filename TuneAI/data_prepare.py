import os
import torch

def generate_turnit_dataset(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    dataset_content = """[ANDROID_MANIFEST]
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.INTERNET" />
    <application android:theme="@style/Theme.TurnIt" android:label="TurnIt">
        <activity android:name=".MainActivity" android:exported="true">
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
}

[COMPOSE_CHAT_UI]
@Composable
fun ChatScreen(messages: List<ChatMessage>) {
    LazyColumn(modifier = Modifier.fillMaxSize()) {
        items(messages) { message ->
            // Gravity.END for User (Right), Gravity.START for AI (Left)
            val alignment = if (message.isUser) Alignment.End else Alignment.Start
            val bgColor = if (message.isUser) Color.Transparent else Color(0x33FFFFFF)
            
            Box(contentAlignment = alignment, modifier = Modifier.fillMaxWidth().padding(8.dp)) {
                // Glassmorphism base
                Surface(shape = RoundedCornerShape(16.dp), color = bgColor) {
                    Text(text = message.text, color = Color.White, modifier = Modifier.padding(12.dp))
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
        initialValue = 0f, targetValue = 360f,
        animationSpec = infiniteRepeatable(animation = tween(2000, easing = LinearEasing))
    )
    Box(modifier = Modifier.drawBehind {
        rotate(angle) {
            // Flowing RGB gradient
            drawRect(brush = Brush.sweepGradient(listOf(Color.Red, Color.Green, Color.Blue, Color.Red)))
        }
    }) { content() }
}

[COMPOSE_NAVIGATION_DRAWER]
@Composable
fun TurnItDrawer(onNavigate: (String) -> Unit) {
    ModalDrawerSheet {
        // 3-line hamburger menu items
        NavigationDrawerItem(icon = { Icon(Icons.Default.Add, "New Chat") }, label = { Text("New Chat") }, selected = false, onClick = { onNavigate("new_chat") })
        NavigationDrawerItem(icon = { Icon(Icons.Default.History, "History") }, label = { Text("History") }, selected = false, onClick = { onNavigate("history") })
        NavigationDrawerItem(icon = { Icon(Icons.Default.Settings, "API Key Settings") }, label = { Text("API Key Settings") }, selected = false, onClick = { onNavigate("settings") })
    }
}
"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(dataset_content.strip())
    print(f"Auto-generated TurnIt dataset at {file_path}")

def prepare_data():
    dataset_dir = 'dataset'
    file_path = os.path.join(dataset_dir, 'llms-full.txt')
    
    # Self-healing logic: Create dataset if it doesn't exist
    if not os.path.exists(dataset_dir) or not any(f.endswith('.txt') for f in os.listdir(dataset_dir)):
        print("Error avoided: No text files found in dataset/. Generating Gold Set...")
        generate_turnit_dataset(file_path)

    text = ""
    for file in os.listdir(dataset_dir):
        if file.endswith('.txt'):
            with open(os.path.join(dataset_dir, file), 'r', encoding='utf-8') as f:
                text += f.read() + "\n"

    if not text:
        raise ValueError("Dataset is empty even after generation attempt.")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    print(f"Data prepared. Vocabulary size: {vocab_size}")
    return text, stoi, itos

if __name__ == "__main__":
    prepare_data()
