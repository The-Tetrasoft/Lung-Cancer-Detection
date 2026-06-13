// Local storage utilities for demo purposes
// This simulates server functionality when the actual server is not available

export interface StoredUser {
  username: string;
  firstName: string;
  lastName: string;
  dateOfBirth: string;
  gender: string;
  email: string;
  phoneNumber: string;
  password: string;
  profilePicture?: string;
  role: string;
  createdAt: string;
  loginAttempts: number;
}

const USERS_KEY = 'lungdx_users';
const EMAILS_KEY = 'lungdx_emails';

// Initialize with admin user
const initializeLocalStorage = () => {
  const users = getStoredUsers();
  const emails = getStoredEmails();
  
  if (!users['Admin']) {
    users['Admin'] = {
      username: 'Admin',
      firstName: 'System',
      lastName: 'Administrator',
      email: 'admin@lungcancer.system',
      password: encryptPassword('CheXNet2025'),
      role: 'admin',
      dateOfBirth: '1980-01-01',
      gender: 'Other',
      phoneNumber: '00000000000',
      createdAt: new Date().toISOString(),
      loginAttempts: 0
    };
    emails['admin@lungcancer.system'] = 'Admin';
    
    localStorage.setItem(USERS_KEY, JSON.stringify(users));
    localStorage.setItem(EMAILS_KEY, JSON.stringify(emails));
  }
};

const getStoredUsers = (): Record<string, StoredUser> => {
  try {
    return JSON.parse(localStorage.getItem(USERS_KEY) || '{}');
  } catch {
    return {};
  }
};

const getStoredEmails = (): Record<string, string> => {
  try {
    return JSON.parse(localStorage.getItem(EMAILS_KEY) || '{}');
  } catch {
    return {};
  }
};

// Simple encryption (same as server)
function encryptPassword(password: string): string {
  const key = "LUNGCANCER2025";
  let encrypted = "";
  for (let i = 0; i < password.length; i++) {
    const charCode = password.charCodeAt(i);
    const keyChar = key.charCodeAt(i % key.length);
    encrypted += String.fromCharCode(((charCode + keyChar) % 256));
  }
  return btoa(encrypted);
}

function decryptPassword(encryptedPassword: string): string {
  const key = "LUNGCANCER2025";
  const encrypted = atob(encryptedPassword);
  let decrypted = "";
  for (let i = 0; i < encrypted.length; i++) {
    const charCode = encrypted.charCodeAt(i);
    const keyChar = key.charCodeAt(i % key.length);
    let originalChar = (charCode - keyChar) % 256;
    if (originalChar < 0) originalChar += 256;
    decrypted += String.fromCharCode(originalChar);
  }
  return decrypted;
}

export const localStorageAPI = {
  // Check username availability
  checkUsername: async (username: string): Promise<{ available: boolean }> => {
    initializeLocalStorage();
    const users = getStoredUsers();
    return { available: !users[username] };
  },

  // Register new user
  signup: async (userData: {
    username: string;
    firstName: string;
    lastName: string;
    dateOfBirth: string;
    gender: string;
    email: string;
    phoneNumber: string;
    password: string;
    profilePicture?: string;
  }): Promise<{ success: boolean; message: string }> => {
    initializeLocalStorage();
    const users = getStoredUsers();
    const emails = getStoredEmails();

    // Check if username exists
    if (users[userData.username]) {
      return { success: false, message: 'Username already exists' };
    }

    // Check if email exists
    if (emails[userData.email]) {
      return { success: false, message: 'Email already registered' };
    }

    // Create new user
    const newUser: StoredUser = {
      ...userData,
      password: encryptPassword(userData.password),
      role: 'user',
      createdAt: new Date().toISOString(),
      loginAttempts: 0
    };

    users[userData.username] = newUser;
    emails[userData.email] = userData.username;

    localStorage.setItem(USERS_KEY, JSON.stringify(users));
    localStorage.setItem(EMAILS_KEY, JSON.stringify(emails));

    return { success: true, message: 'Account created successfully' };
  },

  // User login
  login: async (usernameOrEmail: string, password: string): Promise<{
    success: boolean;
    message: string;
    remainingAttempts?: number;
    user?: any;
  }> => {
    initializeLocalStorage();
    const users = getStoredUsers();
    const emails = getStoredEmails();

    let username = usernameOrEmail;

    // Check if input is email format
    if (usernameOrEmail.includes('@')) {
      const mappedUsername = emails[usernameOrEmail];
      if (!mappedUsername) {
        return { success: false, message: 'No user found, create an account' };
      }
      username = mappedUsername;
    }

    const user = users[username];
    if (!user) {
      return { success: false, message: 'No user found, create an account' };
    }

    // Check if account is locked
    if (user.loginAttempts >= 5) {
      return { success: false, message: 'Account locked due to too many failed attempts' };
    }

    // Verify password
    const decryptedPassword = decryptPassword(user.password);
    if (decryptedPassword !== password) {
      user.loginAttempts = (user.loginAttempts || 0) + 1;
      users[username] = user;
      localStorage.setItem(USERS_KEY, JSON.stringify(users));

      const remainingAttempts = 5 - user.loginAttempts;
      if (remainingAttempts <= 0) {
        return { success: false, message: 'Account locked due to too many failed attempts' };
      }

      return {
        success: false,
        message: 'Incorrect password',
        remainingAttempts
      };
    }

    // Reset login attempts on successful login
    user.loginAttempts = 0;
    users[username] = user;
    localStorage.setItem(USERS_KEY, JSON.stringify(users));

    return {
      success: true,
      message: 'Login successful',
      user: {
        username: user.username,
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email,
        role: user.role,
        profilePicture: user.profilePicture || null
      }
    };
  }
};