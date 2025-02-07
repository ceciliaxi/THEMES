import tensorflow as tf
import random


# Define the Generator (Policy Network)
class Generator(tf.keras.Model):
    def __init__(self, state_dim, action_dim, intention_network):
        super(Generator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.intention_network = intention_network  # Pass the intention network directly
        
        # Hidden layers for the generator
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_dim, activation='softmax')  # Action output with softmax

        # Latent intention projection layer to map intentions to action space
        self.intention_projection = tf.keras.layers.Dense(action_dim, activation='sigmoid')  # Project intentions to action space
        
    def call(self, state, z, latent_intention_probs=None):
        # Ensure the inputs are at least 2D (batch_size, state_dim + z_dim)
        state = tf.expand_dims(state, axis=0) if len(state.shape) == 1 else state
        z = tf.expand_dims(z, axis=0) if len(z.shape) == 1 else z
        
        # Ensure `z` matches the batch size of `state`
        batch_size = tf.shape(state)[0]  # Get the batch size from `state`
        z = tf.tile(z, [batch_size, 1])   # Replicate `z` for each item in the batch
        
        # Combine state and z
        x = tf.concat([state, z], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        action_probs = self.output_layer(x)
        
        # If latent intentions are not passed directly, generate them from z using the intention network
        if latent_intention_probs is None:
            latent_intention_probs = self.intention_network(z)  # Get intention probabilities from the intention network
        
        # Project latent intentions to action space (project intentions to match action space)
        intention_projection = self.intention_projection(latent_intention_probs)  # Shape: [batch_size, action_dim]
        final_policy = action_probs * intention_projection
        
        return final_policy


class Discriminator(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hidden layers for the discriminator
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')  # Output probability that the pair is real
    
    def call(self, state, action):
        # Combine state and action
        x = tf.concat([state, action], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        
        # Output real or fake probability
        return self.output_layer(x)


# Latent Intention Network
class LatentIntentionNetwork(tf.keras.Model):
    def __init__(self, z_dim, num_intentions):
        super(LatentIntentionNetwork, self).__init__()
        self.z_dim = z_dim
        self.num_intentions = num_intentions
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_intentions, activation='softmax')  # Output probabilities for each latent intention
        
    def call(self, z):
        x = self.dense1(z)
        x = self.dense2(x)
        intentions = self.output_layer(x)
        return intentions

def generator_loss(generator, discriminator, state, z, true_actions, latent_intention_probs, weight_H=0.01, weight_I=0.01):
    # Generate the policy
    policy = generator(state, z, latent_intention_probs)
    
    # Cross-entropy loss for action prediction and intention regularization
    action_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(true_actions, policy, from_logits=True))
    
    # Entropy terms
    action_entropy = -tf.reduce_mean(tf.reduce_sum(policy * tf.math.log(policy + 1e-8), axis=-1))  # Entropy of actions
    intention_entropy = -tf.reduce_mean(tf.reduce_sum(latent_intention_probs * tf.math.log(latent_intention_probs + 1e-8), axis=-1))  # Entropy of intentions
    
    # Final generator loss combining these terms
    total_loss = action_loss + weight_H * action_entropy + weight_I * intention_entropy
    
    return total_loss


def discriminator_loss(disc_output_real, disc_output_fake):
    # Cross-entropy loss (real vs. fake)
    epsilon = 1e-8  # Small epsilon to avoid log(0)
    return -tf.reduce_mean(tf.math.log(disc_output_real + epsilon) + tf.math.log(1. - disc_output_fake + epsilon))


# Sampling function for latent variable z
def sample_latent_variable(z_dim):
    return tf.random.normal(shape=(1, z_dim))  # Gaussian random noise

# Batch processing function (to shuffle and batch the expert data)
def get_batches(expert_data, batch_size):
    random.shuffle(expert_data)
    for i in range(0, len(expert_data), batch_size):
        yield expert_data[i:i + batch_size]


# Training loop
def train_info_gail(expert_data, generator, discriminator, optimizer_g, optimizer_d, z_dim,
                    epochs=1000, batch_size=64, print_interval=100):
    for epoch in range(epochs):
        total_g_loss = 0.0
        total_d_loss = 0.0
        
        # Loop through batches of expert data
        for batch in get_batches(expert_data, batch_size):
            batch_states = [item[0] for item in batch]
            batch_actions = [item[1] for item in batch]
            
            # Convert to tensors
            batch_states = tf.convert_to_tensor(batch_states, dtype=tf.float32)
            batch_actions = tf.convert_to_tensor(batch_actions, dtype=tf.float32)

            with tf.GradientTape() as tape:
                # Train the discriminator
                real_action_probs = discriminator(batch_states, batch_actions)  # Real actions from the expert

                z = tf.random.normal([1, z_dim])
                latent_intention_probs = generator.intention_network(z)
                
                generated_actions = generator(batch_states, z, latent_intention_probs)
                fake_action_probs = discriminator(batch_states, generated_actions)  # Fake actions from the generator
                
                # Discriminator loss (real vs fake)
                d_loss = -tf.reduce_mean(tf.math.log(real_action_probs) + tf.math.log(1 - fake_action_probs))
            

            # Update the discriminator's parameters
            grads_d = tape.gradient(d_loss, discriminator.trainable_variables)
            optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))
            
        
            # # Step 2: Train the generator
            with tf.GradientTape() as tape:
                g_loss = generator_loss(generator, discriminator, batch_states, z, batch_actions, latent_intention_probs)

            # Update the generator's parameters
            grads_g = tape.gradient(g_loss, generator.trainable_variables)
            optimizer_g.apply_gradients(zip(grads_g, generator.trainable_variables))
            
            # Accumulate losses
            total_g_loss += g_loss
            total_d_loss += d_loss
        
        # Optional: Print losses at intervals for monitoring
        if epoch % print_interval == 0:
            avg_g_loss = total_g_loss / len(expert_data)
            avg_d_loss = total_d_loss / len(expert_data)
            print(f"Epoch {epoch}: Generator Loss = {avg_g_loss.numpy()}, Discriminator Loss = {avg_d_loss.numpy()}")
    
    return generator, discriminator