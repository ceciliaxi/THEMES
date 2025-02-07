import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

# Define the Generator (Policy Network)
class Generator(tf.keras.Model):
    def __init__(self, state_dim, action_dim, z_dim):
        super(Generator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        
        # Hidden layers for the generator
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_dim, activation='sigmoid')  # Action output

        # Project z to action space (z_dim -> action_dim)
        self.z_projection = tf.keras.layers.Dense(action_dim, activation=None)
        
    def call(self, state, z):
        # Ensure the inputs are at least 2D (batch_size, state_dim + z_dim)
        state = tf.expand_dims(state, axis=0) if len(state.shape) == 1 else state
        z = tf.expand_dims(z, axis=0) if len(z.shape) == 1 else z

        # Ensure `z` matches the batch size of `state`
        batch_size = tf.shape(state)[0]  # Get the batch size from `state`
        z = tf.tile(z, [batch_size, 1])   # Replicate `z` for each item in the batch
        
        # Combine state and latent variable z
        x = tf.concat([state, z], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        action = self.output_layer(x)
        return action

# Define the Discriminator (Critic)
class Discriminator(tf.keras.Model):
    def __init__(self, state_dim, action_dim, z_dim):
        super(Discriminator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        
        # Layers for the discriminator
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')  # Output real/fake (binary)
        
    def call(self, state, action, z):
        # Ensure state, action, and z are at least 2D
        state = tf.expand_dims(state, axis=0) if len(state.shape) == 1 else state
        action = tf.expand_dims(action, axis=0) if len(action.shape) == 1 else action
        z = tf.expand_dims(z, axis=0) if len(z.shape) == 1 else z

        # Ensure `z` matches the batch size of `state`
        batch_size = tf.shape(state)[0]  # Get the batch size from `state`
        z = tf.tile(z, [batch_size, 1])   # Replicate `z` for each item in the batch
        
        # Combine state, action, and z latent variable
        x = self.concat([state, action, z])
        x = self.dense1(x)
        x = self.dense2(x)
        real_fake = self.output_layer(x)
        return real_fake

# # Loss Functions
# def generator_loss(disc_output_fake):
#     # Negative log-likelihood of the discriminator's judgment
#     epsilon = 1e-8  # Small epsilon to avoid log(0)
#     return -tf.reduce_mean(tf.math.log(disc_output_fake + epsilon))

# def generator_loss(disc_output_fake, generated_actions, action_labels):
#     # 1. Log-likelihood of generated actions (maximize the probability of real actions)
#     epsilon = 1e-8  # Small epsilon to avoid log(0)
    
#     # Calculate binary cross-entropy for the generated action probabilities
#     log_likelihood = -tf.reduce_mean(
#         action_labels * tf.math.log(generated_actions + epsilon) +
#         (1 - action_labels) * tf.math.log(1 - generated_actions + epsilon)
#     )
    
#     # 2. Entropy of the current policy (encourages exploration)
#     entropy = -tf.reduce_mean(
#         generated_actions * tf.math.log(generated_actions + epsilon) + 
#         (1 - generated_actions) * tf.math.log(1 - generated_actions + epsilon)
#     )
    
#     # Combine log-likelihood and entropy
#     return log_likelihood - entropy
# 

# # Generator Loss function
# def generator_loss(disc_output_fake, generated_actions, latent_intention_probs, z, lambda_entropy=1.0):
#     epsilon = 1e-8  # Small epsilon to avoid log(0)

#     # 1. Maximizing log(D_w(s,a)) for fake state-action pairs (trying to fool the discriminator)
#     disc_loss = -tf.reduce_mean(tf.math.log(disc_output_fake + epsilon))  # Maximizing disc_output_fake

#     # 2. Log-likelihood of p(i|s,a) (latent intention inference term)
#     log_intention = tf.reduce_sum(latent_intention_probs * tf.math.log(generated_actions + epsilon), axis=-1)

#     # 3. Entropy term for the current policy (encourages exploration)
#     entropy = -tf.reduce_mean(generated_actions * tf.math.log(generated_actions + epsilon) + 
#                               (1 - generated_actions) * tf.math.log(1 - generated_actions + epsilon))

#     # 4. Combine all components of the generator loss
#     total_g_loss = disc_loss + log_intention - lambda_entropy * entropy
#     return total_g_loss 



def generator_loss(disc_output_fake, generated_actions, log_intention, lambda_entropy=0.01):
    epsilon = 1e-8  # To avoid log(0)
    
    # 1. Discrimination loss: maximize the log of discriminator's judgment of fake actions
    g_loss_disc = -tf.reduce_mean(tf.math.log(disc_output_fake + epsilon))  # Maximize fake log likelihood
    
    # 2. Latent intention cost: Encourage actions that make intention inference easier
    g_loss_intention = tf.reduce_mean(log_intention)
    
    # 3. Entropy regularization: Encourage entropy in the policy (promote exploration)
    # Assuming you're computing entropy from the action probabilities (this could be optional)
    # Entropy of the current policy is computed over the generated actions
    policy_entropy = -tf.reduce_mean(tf.reduce_sum(generated_actions * tf.math.log(generated_actions + epsilon), axis=-1))
    
    # Total generator loss = Discrimination loss + Latent intention loss + Entropy regularization
    total_g_loss = g_loss_disc + g_loss_intention + lambda_entropy * policy_entropy
    return total_g_loss



def discriminator_loss(disc_output_real, disc_output_fake):
    # Cross-entropy loss (real vs. fake)
    epsilon = 1e-8  # Small epsilon to avoid log(0)
    return -tf.reduce_mean(tf.math.log(disc_output_real + epsilon) + tf.math.log(1. - disc_output_fake + epsilon))

# Loss Functions
# def mutual_information_loss(z, generated_actions):
#     # Now compute the loss between the projected z and generated actions
#     return tf.reduce_mean(tf.square(z - generated_actions))  # L2 loss between projected z and generated_actions

# Sampling function for latent variable z
def sample_latent_variable(z_dim):
    return tf.random.normal(shape=(1, z_dim))  # Gaussian random noise

# Batch processing function (to shuffle and batch the expert data)
def get_batches(expert_data, batch_size):
    random.shuffle(expert_data)
    for i in range(0, len(expert_data), batch_size):
        yield expert_data[i:i + batch_size]

# Training loop
# def train_info_gail(expert_data, generator, discriminator, optimizer_g, optimizer_d, z_dim, epochs=1000, batch_size=64, print_interval=100, lambda_entropy=0.5):
def train_info_gail(expert_data, generator, discriminator, optimizer_g, optimizer_d, intention_network, z_dim, epochs=1000, batch_size=64, print_interval=100):
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
            
            with tf.GradientTape(persistent=True) as tape:
                # Step 1: Sample a random latent variable z
                z = sample_latent_variable(z_dim)
                
                # Step 2: Generate action from the generator (policy)
                generated_actions = generator(batch_states, z)
                
                # Step 3: Discriminator's judgment on real (expert) and fake (generated) actions
                disc_output_real = discriminator(batch_states, batch_actions, z)
                disc_output_fake = discriminator(batch_states, generated_actions, z)


                # Compute latent intention probabilities from the intention network
                latent_intention_probs = intention_network(z)  #

                # Compute latent intention (this part depends on your actual implementation)
                log_intention = tf.reduce_sum(latent_intention_probs * tf.math.log(generated_actions + 1e-8), axis=-1)
                
                # Calculate the generator loss
                g_loss = generator_loss(disc_output_fake, generated_actions, log_intention, lambda_entropy=0.01)
                
                # Calculate the discriminator loss
                d_loss = discriminator_loss(disc_output_real, disc_output_fake)
            
            # Update the generator's parameters
            grads_g = tape.gradient(g_loss, generator.trainable_variables)
            optimizer_g.apply_gradients(zip(grads_g, generator.trainable_variables))
            
            # Update the discriminator's parameters
            grads_d = tape.gradient(d_loss, discriminator.trainable_variables)
            optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))
            
            # Accumulate losses
            total_g_loss += g_loss
            total_d_loss += d_loss


                # Calculate the latent intention cost
                # # Here, latent_intention_probs is generated from the discriminator output or another source
                # latent_intention_probs = tf.nn.softmax(disc_output_fake, axis=-1)  # Assuming you get probabilities from discriminator
                
                # # Calculate the log of generated actions with a small epsilon to avoid log(0)
                # log_generated_actions = tf.math.log(generated_actions + 1e-8)  # Shape: [batch_size, action_dim]

                # # Expand latent_intention_probs to match the shape of generated_actions for element-wise multiplication
                # latent_intention_probs_expanded = tf.expand_dims(latent_intention_probs, axis=-1)  # Shape: [batch_size, num_intentions, 1]
                # latent_intention_probs_expanded = tf.tile(latent_intention_probs_expanded, [1, 1, tf.shape(generated_actions)[-1]])  # Shape: [batch_size, num_intentions, action_dim]

                # # Multiply latent intention probabilities with log of generated actions
                # log_intention = tf.reduce_sum(latent_intention_probs_expanded * log_generated_actions, axis=-2)  # Shape: [batch_size, action_dim]
                
                # # Step 4: Calculate losses
                # g_loss = generator_loss(disc_output_fake)  # Generator loss (maximize log fake prob)
                # d_loss = discriminator_loss(disc_output_real, disc_output_fake)  # Discriminator loss (real/fake)
                
                # # Total generator loss includes the intention loss and entropy regularization
                # total_g_loss_batch = g_loss + lambda_entropy * tf.reduce_mean(log_intention)
                # total_d_loss_batch = d_loss


                
                # Step 4: Calculate losses
                # g_loss = generator_loss(disc_output_fake)  # Generator loss (maximize log fake prob)
                # latent_intention_probs = tf.nn.softmax(z, axis=-1) 
                # g_loss = generator_loss(disc_output_fake, generated_actions, batch_actions, batch_states, z, batch_actions.shape[-1])
                # latent_intention_probs = tf.nn.softmax(disc_output_fake, axis=-1)  # Assuming you get probabilities from discriminator
                
                # Calculate the log of generated actions with a small epsilon to avoid log(0)
                # log_generated_actions = tf.math.log(generated_actions + 1e-8)  # Shape: [batch_size, action_dim]

                # # Expand latent_intention_probs to match the shape of generated_actions for element-wise multiplication
                # latent_intention_probs_expanded = tf.expand_dims(latent_intention_probs, axis=-1)  # Shape: [batch_size, num_intentions, 1]
                # latent_intention_probs_expanded = tf.tile(latent_intention_probs_expanded, [1, 1, tf.shape(generated_actions)[-1]])  # Shape: [batch_size, num_intentions, action_dim]

                # g_loss = generator_loss(disc_output_fake, generated_actions, latent_intention_probs, z, lambda_entropy)




                # d_loss = discriminator_loss(disc_output_real, disc_output_fake)  # Discriminator loss (real/fake)

                # # Project `z` to match the action space dimension
                # # z_projected = generator.z_projection(z)
                # # mi_loss = mutual_information_loss(z_projected, generated_actions)  # Latent variable loss
                
                # # Total generator loss (includes mutual information loss)
                # total_g_loss_batch = g_loss # + 0.1*mi_loss
                # total_d_loss_batch = d_loss
            
            # Step 5: Update the generator
            grads_g = tape.gradient(total_g_loss_batch, generator.trainable_variables)
            optimizer_g.apply_gradients(zip(grads_g, generator.trainable_variables))
            
            # Step 6: Update the discriminator
            grads_d = tape.gradient(total_d_loss_batch, discriminator.trainable_variables)
            optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))
            
            # Accumulate the losses
            total_g_loss += total_g_loss_batch
            total_d_loss += total_d_loss_batch
        
        # Optional: Print losses at intervals for monitoring
        if epoch % print_interval == 0:
            avg_g_loss = total_g_loss / len(expert_data)
            avg_d_loss = total_d_loss / len(expert_data)
            print(f"Epoch {epoch}: Generator Loss = {avg_g_loss.numpy()}, Discriminator Loss = {avg_d_loss.numpy()}")
    
    return generator, discriminator